# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Canopy integration for verl: tree-aware rollout scheduling on the training side.

Targets verl commit 59f53cc4c92dee9609701bbb10d71238bd9ca86e with vLLM 0.13.0.

  * ``scheduler``     -- tree-aware placement (``TreeAwareLoadBalancer``), the
                         agent-loop manager that installs it, the admission
                         coordinator, and the patch that carries ``tree_uid``
                         through verl's ``AsyncLLMServerManager``.
  * ``vllm_patch``    -- runtime patches of vLLM 0.13 for shared-prefix retention
                         (prefix lock) and prefix-preserving partial preemption,
                         and the RPC methods they add to verl's ``vLLMHttpServer``.
  * ``bootstrap``     -- ``sitecustomize.py`` that applies the engine-side patches
                         in vLLM's engine-core subprocess.
  * ``algorithm``     -- tree metadata shared by tree-rollout agent loops.
  * ``observability`` -- the periodic metrics scrape that ticks the admission
                         coordinator.

Importing this package applies the patches below.  They must be in place
before verl creates its Ray actors: the actor classes are shipped to the
workers when the rollout replicas are constructed, so methods added later never
reach them.  Every patch is idempotent and is skipped when verl or vLLM cannot
be imported.
"""

import os as _os

__all__ = ["algorithm", "observability", "scheduler", "vllm_patch"]

try:
    from canopy_verl.scheduler.patches import apply_patches as _apply_scheduler_patches

    _apply_scheduler_patches()
except Exception:  # verl not importable
    pass

# Shared-prefix retention (admission mode b1).  These patches only add methods and
# an allocation wrapper that is a no-op while no block is locked; behavior changes
# only when the admission coordinator issues lock requests.
try:
    from canopy_verl.vllm_patch import apply_all_b1_patches as _apply_b1_patches

    print(f"[CANOPY-VLLM-PATCH] canopy_verl import-time apply_all_b1_patches results={_apply_b1_patches()}",
          flush=True)
except Exception:  # verl / vLLM not importable
    pass

# Prefix-preserving partial preemption (admission mode b2) wraps vLLM's preemption
# path, so it is applied only when CANOPY_BOOTSTRAP_B2=1.
if _os.environ.get("CANOPY_BOOTSTRAP_B2", "0") == "1":
    try:
        from canopy_verl.vllm_patch import apply_all_b2_patches as _apply_b2_patches

        print(f"[CANOPY-VLLM-PATCH] canopy_verl import-time apply_all_b2_patches results={_apply_b2_patches()}",
              flush=True)
    except Exception:  # verl / vLLM not importable
        pass
