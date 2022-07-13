import json
from pathlib import Path
from argparse import ArgumentParser


def main(args):
    available_tasks = json.load(
        open(
            "/home/khippe/github/lm-evaluation-harness/big_bench_integration/hf_available_tasks.json"
        )
    )
    task_data = json.load(open(args.task_json))

    generate, mc = {}, {}
    for task, data in task_data.items():
        if isinstance(data, dict):
            num_gen = data["num_generate_text"]
            num_mc = data["num_multiple_choice"]

            if num_gen > 0:
                generate[task] = num_gen
            if num_mc > 0:
                mc[task] = num_mc

    # find number of only mc tasks
    only_mc = []
    for key in mc:
        if key not in generate:
            only_mc.append(key)

    available_only_mc = []
    for task in only_mc:
        if task in available_tasks:
            available_only_mc.append(task)

    print(f"Number of mc only tasks: {len(available_only_mc)}")
    json.dump(available_only_mc, open("only_mc_tasks.json", "w"))

    # Find number of generation tasks
    only_gen = []
    for key in generate:
        if key not in mc:
            only_gen.append(key)

    available_only_gen = []
    for task in only_gen:
        if task in available_tasks:
            available_only_gen.append(task)

    print(f"Number of gen only tasks: {len(available_only_gen)}")
    json.dump(available_only_gen, open("only_gen_tasks.json", "w"))

    # Find number of tasks with both
    both = []
    for key in generate:
        if key in mc:
            both.append(key)

    available_both = []
    for task in both:
        if task in available_tasks:
            available_both.append(task)

    print(f"Number of tasks with both mc and gen: {len(available_both)}")
    json.dump(available_both, open("both_gen_mc_tasks.json", "w"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--task_json",
        type=Path,
        required=True,
        help="Path to task_metadata.json from bigbench repo",
    )

    args = parser.parse_args()
    main(args)
