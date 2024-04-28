import argparse


def define():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="default_run",
        help="Name of current experience"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Directory to save results"
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the experiment N times"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed"
    )

    parser.add_argument(
        "--gpuid",
        nargs="+",  # take one or more
        type=int,
        default=[0],
        help="The list of gpu id"
    )

    # =====
    # Dataset
    # =====

    parser.add_argument(
        "--dataset",
        choices=["MNIST"],
        default="MNIST",
        help="Dataset to train on"
    )

    parser.add_argument(
        "--root",
        type=str,
        default="data/",
        help="The root folder of dataset"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="The number samples per batch"
    )

    # =====
    # Training
    # =====

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1000,
        help="Number of epochs"
    )

    return parser


def parse(argv):
    parser = define()
    return parser.parse_args(argv)
