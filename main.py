import argparse

from dotenv import load_dotenv

from src.experiment import run
from src.utils import load_yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    for path in args.config:
        run(load_yaml(path), reset=args.reset)
