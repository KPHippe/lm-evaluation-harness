import json
from pathlib import Path
from argparse import ArgumentParser


def main(args):
    task_data = json.load(open(args.task_json))
    
    generate, mc = {}, {} 
    for task, data in task_data.items(): 
        if isinstance(data, dict):
            num_gen = data['num_generate_text']
            num_mc = data['num_multiple_choice']
            
            if num_gen > 0 : 
                generate[task] = num_gen 
            if num_mc > 0: 
                mc[task] = num_mc 

    only_json = [] 
    for key in mc: 
        if key not in generate: 
            only_json.append(key)

    print(f"Number of json only tasks: {len(only_json)}")
    json.dump(only_json, open('only_json_tasks.json', 'w'))
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--task_json', 
                    type=Path, 
                    required=True,
                    help="Path to task_metadata.json from bigbench repo")

    args = parser.parse_args()
    main(args)