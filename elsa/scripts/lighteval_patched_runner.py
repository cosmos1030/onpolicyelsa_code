"""Thin wrapper around lighteval's CLI entrypoint that patches a known
upstream bug before running, so the fix travels with this repo (via git)
instead of requiring a manual site-packages edit on every machine that runs
evals.

Bug: lighteval.metrics.metrics_sample.AvgAtN.__init__ sets self.n, but
AvgAtN.compute() references self.k (never set), raising
"AttributeError: 'AvgAtN' object has no attribute 'k'" -- this breaks any
task using the avg_at_n_math metric, e.g. the built-in aime24/aime25 tasks.
Confirmed on lighteval as installed in the `rac` conda env; check whether a
newer lighteval release has fixed this before assuming this patch is still
needed.
"""
import os
import sys


def _disable_sample_cache():
    """Opt-out of lighteval's on-disk sample cache (LIGHTEVAL_DISABLE_SAMPLE_CACHE=1).

    Off by default. Kept because every @cached model method round-trips its results
    back OUT of the cache (get_samples_from_cache -> datasets.load_dataset on the
    just-written parquet) before returning them, so the read path runs even on a
    cold cache and clearing the cache does not avoid it -- if that read ever does
    become the problem, this is the switch.

    NOT the fix for the 2026-09-06 hangs, despite being written for them: three
    evals stopped dead just after generation and stayed stopped 40-50 min. Turning
    this on did not help, because the hang is one call further down --
    Pipeline._run_model calls self.model.cleanup() before _compute_metrics, and
    VLLMModel.cleanup (destroy_model_parallel / del self.model / ray.shutdown /
    destroy_distributed_environment) never returns under vLLM V0 multiproc at
    tensor_parallel_size=4 on this host. Every eval that has ever completed here
    ran tp=2 (20/20 across the delta-sweep logs); every tp=4 eval has hung. Run
    evals at tp<=2, or fix cleanup, rather than reaching for this flag.

    Implemented by rebinding cache_management.cached to a pass-through decorator
    BEFORE any model module does `from ... import cached` (the decorator is applied
    at class-definition time, hence the sys.modules guard). The cache OBJECT must
    still exist: Pipeline.__init__ calls self.model._cache._init_registry(...)
    unconditionally, so replacing SampleCache with None instead crashes there.
    """
    if os.environ.get("LIGHTEVAL_DISABLE_SAMPLE_CACHE", "") not in ("1", "true", "True"):
        return

    already = [m for m in sys.modules if m.startswith("lighteval.models.")]
    if already:
        print(f"[lighteval_patched_runner] WARNING: model modules already imported "
              f"({already}); the cache bypass may not take effect", flush=True)

    import functools

    from lighteval.utils import cache_management

    def _passthrough(sampling_method=None):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, docs, *args, **kwargs):
                return func(self, docs, *args, **kwargs)

            return wrapper

        return decorator

    cache_management.cached = _passthrough
    print("[lighteval_patched_runner] sample cache BYPASSED "
          "(LIGHTEVAL_DISABLE_SAMPLE_CACHE=1)", flush=True)


def _patch_avg_at_n():
    import numpy as np
    from lighteval.metrics.metrics_sample import AvgAtN

    def compute(self, doc, model_response, **kwargs):
        all_scores = [self.compute_score(doc, model_response[i]) for i in range(self.n)]
        return np.mean(all_scores)

    AvgAtN.compute = compute


_disable_sample_cache()
_patch_avg_at_n()

from lighteval.__main__ import app  # noqa: E402

if __name__ == "__main__":
    sys.exit(app())
