"""
benchmark — Comparative evaluation of C60.ai against baseline AutoML pipelines.

Runs nested cross-validation (outer 5-fold × 5 seeds) across 4 classification
datasets and reports mean ± std accuracy with Wilcoxon signed-rank significance
tests.
"""
