"""Train one or more experiments and export their split, scores and top-N lists.

Examples:
    python run_model.py UserKNN ItemKNN BPR
    python run_model.py all
    python run_model.py UserKNN --set k 50 --set shrink 10      # quick manual override
    python run_model.py BPR --no-tuned                          # ignore configs/tuned/BPR.yaml
"""

import argparse

import yaml

from common import experiment_names, export, train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+", help=f"'all' or any of: {', '.join(experiment_names())}")
    parser.add_argument("--set", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"),
                        help="override a config value, e.g. --set k 50")
    parser.add_argument("--no-tuned", action="store_true", help="ignore configs/tuned/<experiment>.yaml")
    parser.add_argument("--top-n", type=int, default=100, help="length of the exported recommendation lists")
    args = parser.parse_args()

    experiments = experiment_names() if args.experiments == ["all"] else args.experiments
    overrides = {key: yaml.safe_load(value) for key, value in args.set}

    for experiment in experiments:
        outputs = train(experiment, overrides, use_tuned=not args.no_tuned)
        export(experiment, *outputs, top_n=args.top_n)


if __name__ == "__main__":
    main()
