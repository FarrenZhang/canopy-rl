# Reproducing the paper

This guide maps the serving results of the paper (Figure 2, Section 4, and Appendices D and
E) to the
experiments in [`configs/experiments.json`](../configs/experiments.json) and to the scripts
that print their numbers. Figure, table, and section numbers refer to the camera-ready
version of the paper.

## Hardware and software

- NVIDIA H20 GPUs (96 GB), one vLLM instance per GPU (Qwen3-32B: two GPUs per instance).
- A development build of vLLM 0.20.1 (0.20.1.dev615) and Qwen3 models from Hugging Face.
- Engine limits as in the paper: at most 256 sequences and 8,192 batched tokens per
  iteration, and 90% of GPU memory for weights and KV cache. These are the runner's defaults.
- KV capacity with Qwen3-8B: 31,167 blocks of 16 tokens per instance. Speedups depend on
  memory pressure, so on GPUs with more memory, add `-- --num-gpu-blocks 31167` to
  `run_experiment.py` for the Qwen3-8B experiments; not for `pr_*`, whose reduced capacity it
  would override, nor for `gen_*`, which use other models.
- `sc_repl_r16` and `sc_repl_r32` span several nodes and need a running Ray cluster with 16
  or 32 GPUs. All other experiments run on one node. Multi-node runs report correct times but
  undercount the engines' statistics, so the paper gives no prefill numbers for them.
- Most jobs take under two hours.

## How an experiment runs

Model outputs change from run to run, so to compare policies on identical work the runner
generates a rollout step from a seed as a request graph with fixed inputs, output lengths,
tool latencies, and dependencies, and replays it on the same vLLM servers under each policy.
A turn's input is the previous input plus the tokens the engine actually generated and the
tool output, so prefix-cache hits are those of a live rollout, and every policy issues the
same requests, with the same input and output lengths, in the same dependency order. Each
request asks for exactly its output length. Before each run the prefix caches are reset; an
unmeasured warm-up comes first (long-trajectory jobs with suffix `c`, `f`, `p`, or `v` skip
it), and the policy order rotates across repetitions.

A workload is a JSON object. `kind` is `single` (one-turn groups), `multiturn`, `tree` (with
`fanout`, `branch_at`, and `branch_mode`: `turn` or `inflight`), or `mixed` (a list of
`parts`). Common keys: `groups`, `n` (trajectories or leaves per group), `prefix` (shared
prompt tokens), `turns`, `obs` (tool-output tokens), `gen` (output tokens), `tool_ms` (tool
latency). A value is a number, `[lo, hi]` (uniform integer), or
`{"lognorm": [median, sigma, lo, hi]}`.

Each experiment writes `results/<experiment>/results.json`. Per run it records the rollout
time (`makespan_s`), the prefill computed by the engines (`prefill_computed_tokens`),
preemptions, the KV usage over time, the hook counters (`engine_totals`), the scheduler
counters (`sched_stats`), and for each request its instance, start, end, and input length
(`req_rec`). `workload_stats` describes the generated step.

## Running experiments

```bash
python scripts/run_experiment.py --list                  # all experiments, with their GPUs
python scripts/run_experiment.py m_tree                  # one experiment
python scripts/run_experiment.py 'abr_*'                 # a family
python scripts/run_experiment.py m_tree --model /path/to/Qwen3-8B   # local weights
python scripts/run_experiment.py m_tree --dry-run        # print the runner command
```

`run_experiment.py` gives each experiment the number of GPUs it lists (the first visible
ones) and passes arguments after `--` on to `canopy.bench.runner`. The summaries:

| Script | Prints |
|---|---|
| `scripts/analyze.py [PREFIX ...]` | median rollout time and range per policy, speedup over `vanilla`, prefill as a multiple of the minimum, preemptions; flags runs whose work differs across policies |
| `scripts/prefill_breakdown.py [PREFIX ...]` | prefill duplicated on another instance and recomputed after eviction or preemption, as multiples of the minimum |
| `scripts/admission_wait.py [PREFIX ...]` | admission delay (time beyond the tool latency), when the last trajectory started, preemptions, and the peak of shared-prefix references |
| `scripts/analyze_longtraj.py` | the long-trajectory experiments: speedup over TreeSticky and over verl, and prefill |

From the `m_*` results, these three scripts compute Table 5 (except its Cap column), Table 4,
and Table 7. The quick start in the
[README](../README.md#quick-start) is a short check of the setup.

## Figures and tables

| Paper | What it shows | Experiments | Jobs × GPUs | Summary |
|---|---|---|---|---|
| Figure 6, Table 5 | main results on eight instances | `m_*` | 16 × 8 | `analyze.py m_` |
| Figure 2, Table 4 | where the prefill goes | `m_*` | (same runs) | `prefill_breakdown.py m_` |
| Table 1 | workload characteristics | `m_*` | (same runs) | `workload_stats` in the results |
| Table 7 | admission delay and retained KV | `m_*` | (same runs) | `admission_wait.py m_` |
| Table 6; Cap column of Table 5 | static caps | `cap_*` | 9 × 8 | `analyze.py cap_` |
| Figure 7a | retention mechanisms at 100%, 50%, and 25% of KV memory | `abr_*`, `pr_*`; Canopy at 100% from `abe_*` | 6 × 4, 6 × 4 | `analyze.py abr_ pr_ abe_` |
| Figure 7b | admission with one part removed | `abe_*` | 8 × 4 | `analyze.py abe_` |
| Figure 12 | memory pressure | `abr_*`, `abe_*`, `pr_*` | (same runs) | `analyze.py abr_ abe_ pr_` |
| Table 8 | Canopy against Canopy + both (with shared-prefix references and prefix-preserving preemption) on eight instances | `fx_*` | 13 × 8 | `analyze.py fx_` |
| Section 4.4, Tables 12 and 13 | long trajectories that branch late | `lp_lc*` | 78 × 4 | `analyze_longtraj.py` |
| Figure 8a | number of instances | `sc_repl_r*` | 6 jobs of 1–32 GPUs | `analyze.py sc_repl` |
| Figure 8b | group size | `sc_group_n*` | 8 × 4 | `analyze.py sc_group` |
| Figure 8c | trajectories per GPU | `sc_conc_c*` | 7 × 1 | `analyze.py sc_conc` |
| Figure 8d | tool latency | `sc_tool_t*` | 5 × 1 | `analyze.py sc_tool` |
| Figure 8e | shared-prefix length | `sc_share_p*` | 6 × 4 | `analyze.py sc_share` |
| Section 4.6 | estimate error, λ (the share of KV capacity that admission may fill), estimated output lengths | `rb_prior_*`, `rb_lam_*`, `ga_*` | 2 × 4, 2 × 4, 6 × 4 | `analyze.py rb_ ga_` |
| Table 9 | other models | `gen_*` | 8 × 8 | `analyze.py gen_` |
| Table 10 | mixed tree sizes | `mx_*` | 2 × 8 | `analyze.py mx_` |
| Table 11 | a re-implementation of ThunderAgent's released scheduler | `tf_*` | 9 × 8 | `analyze.py tf_` |
| Figure 9 | learning curves of ARPO training | [`integrations/verl`](../integrations/verl) | 16 GPUs per run | Appendix G |

Some numbers combine experiments: the Cap column of Table 5 scales the best cap's time by
the ratio of Canopy's times in the cap job and in the main job; the last column of Table 11
scales the released scheduler's time in the same way to compare it with verl; and the values
in parentheses in Table 10 come from the main tree runs and their static caps. For Table 1, `workload_stats` gives
the input, output, minimum prefill, and reusable share (`input_tokens`, `gen_tokens`,
`min_prefill_tokens`, `reuse_ratio_input`); the share of the trajectory's own history is not
computed by these scripts.

## Names in the configurations

Workloads (Table 1 and Appendix B):

| Name | Workload in the paper |
|---|---|
| `tree` | Tree search (turn): trees that branch at a turn boundary (Tree-GRPO, AT²PO style) |
| `inflight` | Tree search (in-turn): trees that branch inside a turn, after a tool result (ARPO, AEPO style) |
| `rag` | Local search (GRPO) |
| `api` | Web search (GRPO) |
| `lc` | Long context (GRPO) |
| `swe`, `swelong` | SWE agent (short), SWE agent (long) |
| `math` | Math (control) |

Experiment names are `<family>_<workload>` with an optional suffix. `_r<k>` is the k-th
repetition when a workload ran one repetition per job; `analyze.py` merges these. The
families: `m` main runs, `cap` static caps, `fx` Canopy against Canopy + both on eight
instances (it does not run `canopy_fx`), `abr` retention mechanisms, `pr_<workload>_kv50`
and `_kv25` reduced KV memory, `abe` admission ablation, `rb_prior` and `rb_lam` estimate
error and λ, `ga_<workload>_mean` and `_max` estimated output lengths,
`gen_<model>_<workload>` other models (`q4b`, `q14b`, `q32b`: Qwen3-4B, 14B, 32B; `moe`:
Qwen3-30B-A3B), `mx` mixed tree sizes, `tf` the re-implementation of ThunderAgent's released
scheduler. In the scaling families the suffix is the varied value:
`sc_repl_r<N>` N instances, `sc_group_n<N>` group size, `sc_conc_c<N>` trajectories per GPU,
`sc_tool_t<ms>` median tool latency, `sc_share_p<tokens>` shared-prefix length.

The long-trajectory experiments are `lp_lc<variant>_r<k><s>`. The variants:

| Variant | Trajectory | Branches | Shared prefix |
|---|---:|---:|---:|
| `C` | 93k | 8 | 79k (85%) |
| `L64` | 64k | 8 | 54k (85%) |
| `L128` | 122k | 8 | 103k (84%) |
| `B4` | 94k | 4 | 79k (85%) |
| `B16` | 93k | 16 | 78k (84%) |
| `R75` | 94k | 8 | 70k (75%) |
| `R95` | 98k | 8 | 93k (95%) |
| `R50B4` | 95k | 4 | 46k (49%) |
| `R30B4` | 95k | 4 | 30k (32%) |

Each repetition `k` ran as several jobs; `s` names the policies of a job, and every job
except `c` also runs `sticky` as its baseline: `a` Canopy with the first-request charge
(`canopy`) and Canopy + both (`canopy_full`), `b` Canopy w/o admission (`rv`) and its strict
variant (`rv_hold`), `c` static caps (`stickycap1`–`3`), `f` Canopy (`canopy_fx`), `p`
prefix-aware routing (`par`), `v` verl (`vanilla`). These jobs use a 131,072-token context
with Qwen3-8B's position limit raised through `--hf-overrides`; positions beyond the model's
40,960-token window are extrapolated, and outputs are discarded, so only time and memory
matter.

Policies (values of `--policies`; `POLICIES` in `canopy/bench/runner.py` defines them):

| Policy | In the paper |
|---|---|
| `vanilla` | verl: each trajectory to the instance with the fewest in-flight requests |
| `par` | Prefix-aware routing (the SGLang router's rules) |
| `ttl` | TTL: a trajectory's KV held for 2 s after each turn |
| `sticky` | TreeSticky: each group on one instance |
| `sticky_hint` | TreeSticky with finished-first eviction |
| `rv_nopp`, `rv_nopin`, `rv` | TreeSticky with finished-first eviction plus shared-prefix references, plus prefix-preserving preemption, plus both (= Canopy w/o admission) |
| `rv_hold` | + both, strict: references yield only to running requests |
| `rv_noplace` | Canopy w/o admission and without group placement (not reported in the paper) |
| `thunder` | Program-level: ThunderAgent's policy as re-implemented for the main results |
| `thunder_orig` | Released: a re-implementation of ThunderAgent's released scheduler, with its defaults |
| `thunder_prior` | Program-level with Canopy's previous-step estimate |
| `thunder_l08`, `thunder_l09`, `thunder_l12` | Program-level with λ = 0.8, 0.9, 1.2 |
| `stickycap{K}` | Static cap: at most K root trajectories per instance |
| `canopy` | Canopy (main runs: a group's *b* is taken from its first request) |
| `canopy_fx` | Canopy charging each request its own *b* (Section 4.4) |
| `canopy_full` | Canopy + both: with shared-prefix references and prefix-preserving preemption |
| `canopy_noprior`, `canopy_noshare`, `canopy_noplace`, `canopy_nohint` | Canopy without the previous-step estimate, without counting the shared prefix once, without group placement, without finished-first eviction |
| `canopy_ps025`, `canopy_ps05`, `canopy_ps2`, `canopy_ps4` | Canopy with the previous-step estimate scaled by 0.25, 0.5, 2, 4 |
| `canopy_peak` | Canopy with the estimate taken from the mean per-trajectory peak (not reported in the paper) |
| `canopy_l08`, `canopy_l09` | Canopy with λ = 0.8, 0.9 |

## Not included

- Plotting scripts; the scripts above print the numbers behind each figure.
- The raw results of the paper's runs, and the trajectory samples behind Table 3.
- The correctness check of Appendix B, the runs of Preble, SGLang, and KVFlow on their own
  back ends (Appendix C), and the KV-transfer measurements of Appendix F.
- The agent loop and launch scripts of the ARPO training runs (Figure 9). Appendix G lists
  their settings; [`integrations/verl`](../integrations/verl) holds the integration they used.
