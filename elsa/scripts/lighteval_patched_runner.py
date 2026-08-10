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
import sys


def _patch_avg_at_n():
    import numpy as np
    from lighteval.metrics.metrics_sample import AvgAtN

    def compute(self, doc, model_response, **kwargs):
        all_scores = [self.compute_score(doc, model_response[i]) for i in range(self.n)]
        return np.mean(all_scores)

    AvgAtN.compute = compute


_patch_avg_at_n()

from lighteval.__main__ import app  # noqa: E402

if __name__ == "__main__":
    sys.exit(app())
