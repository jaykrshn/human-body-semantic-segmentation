import argparse


def main(args):
    a = args.number**2
    print(f"The square of your number is {a}")
    print(f"Epochs to be trained: {args.epochs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Returns the square of the given number.",
        usage="python main.py --number <number> [--epochs <epochs>]",
    )
    parser.add_argument(
        "-N",
        "--number",
        type=int,
        help="Number to be squared",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Epochs to be trained, defaults to 1",
    )
    args = parser.parse_args()

    main(args)
