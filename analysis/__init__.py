"""One-shot measurements and diagnostics.

Nothing here is imported by the training or RQ pipeline. Each module answers one question
about the model or the data, prints a verdict, and its finding is recorded in
results/results_summary.csv -- which is what lets a finished one be deleted.

Run them from the repository root: `python analysis/<name>.py --help`.
"""
