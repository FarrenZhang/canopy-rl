# Changes to verl

`canopy_verl` runs on top of an unmodified verl checkout for routing and
admission, but the training experiments also used changes to a few verl files.
This page lists them by file. It describes the changes; it does not ship
modified copies of the verl sources.

Base: verl commit `59f53cc4c92dee9609701bbb10d71238bd9ca86e`
(<https://github.com/volcengine/verl>). verl is licensed under Apache-2.0. If
you distribute modified versions of these files, keep their copyright headers
(listed per file) and add a notice that you modified them.

| File | Needed for | Status |
|---|---|---|
| `verl/experimental/agent_loop/agent_loop.py` | one `tree_uid` per prompt; agent loops that return several trajectories | required (part 1), optional (parts 2-3) |
| `verl/experimental/agent_loop/tool_agent_loop.py` | agent loops that store state in `extra_fields` before the first generate call | recommended |
| `verl/trainer/ppo/ray_trainer.py` | agent loops that return several trajectories | optional |
| `verl/experimental/fully_async_policy/fully_async_rollouter.py` | fully-async trainer with a custom agent-loop manager | optional |
| `verl/experimental/fully_async_policy/fully_async_trainer.py` | rollout wall-clock measurements | optional |
| `tests/experimental/agent_loop/test_agent_loop_extra_fields_schema_on_cpu.py` | tests for part 3 of `agent_loop.py` | optional |

## `verl/experimental/agent_loop/agent_loop.py`

Copyright 2024 Bytedance Ltd. and/or its affiliates.

1. **Trajectory identity for agent loops (required).** In
   `AgentLoopWorker._run_agent_loop`, pass the trajectory info to the agent
   loop as extra keyword arguments:

   ```python
   raw_output = await agent_loop.run(
       sampling_params,
       _canopy_sample_index=trajectory.get("sample_index"),
       _canopy_rollout_n=trajectory.get("rollout_n"),
       _canopy_validate=bool(trajectory.get("validate", False)),
       **kwargs,
   )
   ```

   `canopy_verl.algorithm.derive_shared_tree_uid(_canopy_sample_index)` then
   gives all `rollout.n` coroutines of a prompt the same `tree_uid`, so the
   load balancer places the whole group on one server. Agent loops that do not
   use these arguments ignore them through `**kwargs`.

2. **Several trajectories per coroutine (optional).** Needed only by agent
   loops that keep side branches as extra training samples.
   - Add `class AgentLoopMultiOutput(BaseModel)` with a single field
     `outputs: list[AgentLoopOutput]` (`outputs[0]` is the main trajectory).
     Define it **after** the complete `AgentLoopOutput` class, i.e. after
     `AgentLoopOutput.as_dict`. Inserting it between the fields of
     `AgentLoopOutput` and its `as_dict` method moves `as_dict` to the new
     class and breaks `verl/trainer/main_ppo_sync.py`.
   - `_run_agent_loop` returns a list: one post-processed output for an
     `AgentLoopOutput`, one per element for an `AgentLoopMultiOutput`.
   - `AgentLoopWorker.generate_sequences` flattens the lists and records, for
     every output, the index of the input row it came from (`origin_indices`).
     `_postprocess(..., origin_indices=...)` expands `input_non_tensor_batch`
     to the output rows with `arr[origin_indices]` when the counts differ.
     The worker stores the indices in the non-tensor column
     `canopy_origin_idx_local` (not in `meta_info`, which cannot be
     concatenated across workers).
   - `AgentLoopManager.generate_sequences` adds each worker's chunk offset to
     `canopy_origin_idx_local` and stores the result as
     `canopy_origin_idx_global` before `DataProto.concat`.

   With single-output agent loops the indices are the identity and nothing
   changes.

3. **Padding of responses (recommended).**
   - Pad `response_mask` with a small helper (`_pad_response_mask`: flatten,
     truncate to `response_length`, right-pad with zeros) instead of
     `tokenizer.pad`.
   - Truncate `response_logprobs` to `response_length` before padding.
   - In `_postprocess`, emit `rollout_log_probs` if any output has log-probs,
     filling zeros for outputs without them.

## `verl/experimental/agent_loop/tool_agent_loop.py`

Copyright 2025 Bytedance Ltd. and/or its affiliates.

Detect the first generate call with
`if "min_global_steps" not in agent_data.extra_fields:` instead of
`if not agent_data.extra_fields:`. Agent loops that put their own entries into
`extra_fields` before the first generate call otherwise skip copying the
server's `extra_fields`, and the fully-async trainer later fails on the missing
`min_global_steps`. For plain `ToolAgentLoop` both conditions are equivalent.

## `verl/trainer/ppo/ray_trainer.py`

Copyright 2024 Bytedance Ltd. and/or its affiliates; Copyright 2023-2024 SGLang
Team; Copyright 2025 ModelBest Inc. and/or its affiliates.

Only for agent loops that return several trajectories (part 2 above):

- Training step: after `batch.repeat(repeat_times=rollout.n, interleave=True)`,
  if `gen_batch_output.non_tensor_batch["canopy_origin_idx_global"]` has a
  different length than `batch`, reorder with
  `batch = batch.select_idxs(canopy_origin_idx_global)` before
  `batch.union(gen_batch_output)`.
- In that case, pad the batch to a multiple of
  `dp_size * actor.ppo_mini_batch_size` with
  `verl.protocol.pad_dataproto_to_divisor`. Padding duplicates rows, and the
  duplicates enter the group statistics of GRPO, which biases the advantage
  estimate slightly. Prefer settings where the number of trajectories is
  already divisible.
- Validation: apply the same reordering to `test_batch` before
  `test_batch.union(test_output_gen_batch)`.

## `verl/experimental/fully_async_policy/fully_async_rollouter.py`

Copyright 2025 Meituan Ltd. and/or its affiliates.

- In `_init_async_rollout_manager`, load
  `actor_rollout_ref.rollout.agent.agent_loop_manager_class` with
  `verl.utils.import_utils.load_class_from_fqn` when it is set, as
  `ray_trainer.py` already does; otherwise use `FullyAsyncAgentLoopManager`.
  The class must extend `FullyAsyncAgentLoopManager`. A fully-async variant of
  `CanopyAgentLoopManager` is not part of this release.
- Optional: `async_training.max_concurrent_samples_override` (default `null`)
  replaces the computed `max_concurrent_samples` when positive.
- Optional: `get_tree_counts(samples_per_tree)` returns
  `total_generated_samples // samples_per_tree` (completed trees) and
  `dropped_stale_samples // samples_per_tree` for rollout-throughput accounting.

## `verl/experimental/fully_async_policy/fully_async_trainer.py`

Copyright 2025 Meituan Ltd. and/or its affiliates.

Measurement only: accumulate a decomposition of rollout wall-clock time during
`fit()` and write it to `<run_dir>/wall_clock_decomposition.json` at the end,
together with the tree counts from `get_tree_counts`.

## `tests/experimental/agent_loop/test_agent_loop_extra_fields_schema_on_cpu.py`

New CPU tests for part 3 of `agent_loop.py`: `_pad_response_mask` flattens
and right-pads the mask, and `_postprocess` zero-fills missing log-probs in a
batch where only some outputs have them.
