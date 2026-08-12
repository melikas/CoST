"""GLOBEM entry point.

Thin wrapper: it only flips the defaults that GLOBEM needs and hands over to
``train_hrd.main``. All the training, probing and reporting is the HRD code -- there is
exactly one copy of it, so a fix to the sweep reaches both datasets.

    python train_globem.py --sensor-csv datasets/GLOBEM_REDUCED.csv --output-dir results_globem

Anything ``train_hrd.py`` accepts works here too and wins over the defaults below.
"""
import sys

import train_hrd

# GLOBEM is segment-level (4/day), so the HRD defaults for window length, bin width and
# backbone are wrong for it. Injected as argv so a user-supplied value still overrides.
DEFAULTS = {
    "--dataset": "globem",
    "--sensor-csv": "datasets/GLOBEM_REDUCED.csv",
    "--window-days": "28",
    "--stride-days": "7",
    "--output-dir": "results_globem",
}


def main() -> None:
    argv = sys.argv[1:]
    for flag, value in DEFAULTS.items():
        if flag not in argv:
            argv += [flag, value]
    sys.argv = [sys.argv[0]] + argv
    train_hrd.main()


if __name__ == "__main__":
    main()
