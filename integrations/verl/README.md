# Canopy for verl

This directory holds the integration of Canopy into the
[verl](https://github.com/volcengine/verl) RL trainer that was used for the
training experiments in the paper (the learning curves). It targets:

- verl commit `59f53cc4c92dee9609701bbb10d71238bd9ca86e`
- vLLM `0.13.0` (V1 engine)

It is an earlier implementation than the engine hooks in `canopy/engine/`
(vLLM 0.20), which the rollout-replay benchmark uses. It covers three of the
four mechanisms:

| Mechanism | Implementation here |
|---|---|
| Group-level placement | `TreeAwareLoadBalancer` sends all requests with the same `tree_uid` to one vLLM server; a request spills to the least-loaded server only when its server holds more than 75% (`sticky_threshold`) of the in-flight requests and leads the least-loaded server by more than 10 (`abs_spill_gap`). |
| KV retention of shared prefixes | Mode `b1`: the KV blocks of each tree's latest in-flight requests are locked in vLLM's block pool for a TTL, so eviction skips them. |
| Prefix-preserving preemption | Mode `b2` (includes `b1`): under KV pressure, each tree's latest in-flight requests are tagged; when vLLM preempts a tagged request it keeps the first `partial_preempt_keep_n_blocks` blocks (8 by default, as in the paper's runs) and frees the rest. |
| Group-level admission | Not part of this integration (see the `canopy` policy of the benchmark). |

The modes `a`, `b1` and `b2` (`canopy_scheduler.admission.mode`) name settings of this
integration's retention coordinator (`AdmissionCoordinator`), not the paper's group-level
admission.

## Contents

```
canopy_verl/
  __init__.py                    applies the patches when imported
  scheduler/placement.py         TreeAwareLoadBalancer
  scheduler/patches.py           carries tree_uid through AsyncLLMServerManager.generate
  scheduler/canopy_agent_loop_manager.py   CanopyAgentLoopManager (installs the above)
  scheduler/admission.py         AdmissionCoordinator (modes a / b1 / b2)
  vllm_patch/                    vLLM 0.13 patches and vLLMHttpServer RPC methods
  bootstrap/sitecustomize.py     applies engine-side patches in vLLM's engine-core process
  observability/prom_hook.py     maybe_scrape(): scrapes vLLM metrics, ticks the coordinator
  algorithm/tree_metadata.py     TreeState, derive_shared_tree_uid
  configs/                       canopy_ppo_trainer.yaml, canopy_scheduler.yaml
  tests/                         CPU unit tests
patches/README.md                changes to verl itself
```

## Installation

```bash
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 59f53cc4c92dee9609701bbb10d71238bd9ca86e
# Apply the changes described in patches/README.md (at least part 1 of agent_loop.py).
pip install -e .                 # not the [vllm] extra: it pins vllm<=0.12.0
pip install vllm==0.13.0 "tensordict>=0.8.0,<=0.10.0,!=0.9.0"
```

## Enabling it

**1. Environment of every Ray worker.** `canopy_verl` must be importable in
all workers, and the bootstrap directory must be on `PYTHONPATH` so that
vLLM's engine-core subprocess loads `sitecustomize.py`:

```bash
export PYTHONPATH=<canopy>/integrations/verl/canopy_verl/bootstrap:<canopy>/integrations/verl:$PYTHONPATH
export CANOPY_BOOTSTRAP_B1=1     # admission modes b1 and b2
export CANOPY_BOOTSTRAP_B2=1     # admission mode b2
```

Export these before starting a local Ray instance, or put them in the
`runtime_env.env_vars` of the Ray job.

**2. Trainer configuration.** Use the provided overlay (run from the verl
checkout, because its search path is relative):

```bash
python3 -m verl.trainer.main_ppo \
    --config-path=<canopy>/integrations/verl/canopy_verl/configs \
    --config-name=canopy_ppo_trainer \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=<your agent_loops.yaml> \
    actor_rollout_ref.rollout.agent.default_agent_loop=<your agent loop> \
    data.train_files=<train.parquet> data.val_files=<val.parquet>
```

or add the equivalent overrides to an existing command:

```
actor_rollout_ref.rollout.agent.agent_loop_manager_class=canopy_verl.scheduler.canopy_agent_loop_manager.CanopyAgentLoopManager
actor_rollout_ref.rollout.prometheus.enable=true
actor_rollout_ref.rollout.disable_log_stats=false
+canopy_scheduler.placement.enabled=true
+canopy_scheduler.admission.enabled=true
+canopy_scheduler.admission.mode=b2
```

`configs/canopy_scheduler.yaml` lists all options with their defaults. For the
smaller configurations use `canopy_scheduler.admission.enabled=false`
(placement only) or `canopy_scheduler.admission.mode=b1` (placement and
retention). With `placement.enabled=false` and `admission.enabled=false` the
routing is identical to verl's.

**3. Agent loop.** The agent loop must tell the load balancer which tree a
request belongs to, and it must call `maybe_scrape`, which ticks the admission
coordinator (the coordinator has no timer of its own). For example, on top of
verl's `ToolAgentLoop`:

```python
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

import canopy_verl  # noqa: F401  (applies the patches in this worker)
from canopy_verl.algorithm import derive_shared_tree_uid
from canopy_verl.observability.prom_hook import maybe_scrape
from canopy_verl.scheduler.patches import set_tree_uid_for_routing


class CanopyToolAgentLoop(ToolAgentLoop):
    async def run(self, sampling_params, **kwargs):
        # One tree per prompt: all rollout.n coroutines share the tree_uid.
        self._tree_uid = derive_shared_tree_uid(kwargs.get("_canopy_sample_index"))
        return await super().run(sampling_params, **kwargs)

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        set_tree_uid_for_routing(self._tree_uid)  # per asyncio task
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        await maybe_scrape(agent_data.assistant_turns, algo_tag="TOOL", server_manager=self.server_manager)
        return state
```

Register the class for verl's agent-loop workers with an
`agent_loop_config_path` file (a decorator in the driver does not reach the
Ray workers):

```yaml
- name: canopy_tool_agent
  _target_: my_package.my_module.CanopyToolAgentLoop
```

`_canopy_sample_index` is passed by the `agent_loop.py` change in
`patches/README.md`. Without it, `derive_shared_tree_uid(None)` returns a new
id per coroutine and placement only keeps the turns of one trajectory together.
A loop that calls `server_manager.generate` itself can pass `tree_uid=...`
instead of setting the ContextVar.

## Checking that it is active

Look for these lines in the logs:

```
[canopy-placement] AsyncLLMServerManager patched for tree_uid-aware routing
[CANOPY-PLACEMENT] TreeAwareLoadBalancer initialised enabled=True ...
[CANOPY-ADMISSION] AdmissionCoordinator initialised enabled=True mode=b2 ...
[CANOPY-BOOTSTRAP pid=<PID>] BlockPool PATCHED via sitecustomize meta_path hook
[CANOPY-BOOTSTRAP pid=<PID>] EngineCore PATCHED (utility methods attached)
[PROMHOOK-ADMISSION] algo=... total_ticks=1 ...
```

`maybe_scrape` returns the coordinator's `admission/*` counters after each
tick, and `load_balancer.get_metrics.remote()` returns the `placement/*`
counters (for example `placement/siblings_co_located_ratio`).

Other environment variables: `CANOPY_PROM_RATE` (scrape every N-th call per
worker, default 10) and `CANOPY_ADMISSION_TICK_TIMEOUT` (seconds, default 5).

## Known limitations

- `TreeAwareLoadBalancer.on_tree_complete` is not called by this package.
  Tree registrations therefore stay in the balancer (one per distinct
  `tree_uid`), and prefix locks are released by their TTL. Call
  `on_tree_complete.remote(tree_uid)` on the balancer when a tree is done.
- With vLLM's default multiprocess engine, the bootstrap installs the prefix
  lock and the partial-preemption tag utilities inside the engine-core
  process, while the `Scheduler._preempt_request` wrap is installed by
  `apply_partial_preempt_patch()` in processes that import `canopy_verl`.
  Check that `admission/partial_preempt_preempt_invocations` (mirrored from
  the engine core) increases when vLLM preempts; if it stays at 0, the wrap is
  not active in the engine core. `admission/partial_preempt_succeeded` counts
  accepted tag requests, not partial preemptions.
- Mode `b1` locks all blocks currently held by a tree's latest in-flight
  request(s), which includes but is not limited to the shared prefix.
- The coordinator mirrors lock and preemption counters from the first server
  that answers, and `maybe_scrape` reads the KV usage of the first server known
  to the worker; with several replicas both describe a single replica.
- Only the synchronous trainer is covered (`CanopyAgentLoopManager` extends
  `AgentLoopManager`). A manager for verl's fully-async trainer is not included.
- Admission mode `a` (dummy keepalive requests, optional cascade abort) is not
  part of the evaluated system.

## Tests

```bash
cd integrations/verl
python -m pytest canopy_verl/tests
```

vLLM and verl are replaced by stubs. `test_placement.py` and
`test_admission_coordinator.py` start a local CPU-only Ray instance (they are
skipped if Ray is not installed; the balancer also needs `cachetools`); the
other tests need only pytest.
