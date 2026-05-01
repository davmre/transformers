import dataclasses
from argparse import ArgumentParser
from pathlib import Path

from daves_transformer_lib.training.training import RunSpec, load_config, run_training


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Directory to save run checkpoints and logs.",
    )
    # Config file args are parsed by `config_parser` above, but add redundant
    # definitions here so they appear in `--help`.
    # add_config_args(parser)

    args = parser.parse_args()

    config = load_config(args.config)
    assert len(config.runs) == 1
    run_spec: RunSpec = config.runs[0]

    print("Model config:")
    print(run_spec.model_config)
    print("Training config")
    print(run_spec.training_config)

    rs = dataclasses.replace(run_spec, explicit_run_dir=args.run_dir)
    rs = dataclasses.replace(
        rs, explicit_run_dir=rs.run_dir(base_dir=Path("runs")).absolute()
    )
    run_training(run_spec=rs)


if __name__ == "__main__":
    main()
