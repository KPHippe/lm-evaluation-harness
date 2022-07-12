import json
from pathlib import Path
from argparse import ArgumentParser


def main(args):
    task_data = json.load(open(args.task_json))

    print(task_data.keys())
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--task_json', type=Path, required=True,help="Path to task_metadata.json from bigbench repo")

    args = parser.parse_args()
    main(args)