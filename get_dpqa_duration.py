
import json
import os.path
from animation import CodeGen
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help='directory of SMT results', type=str)
parser.add_argument('--out_dir', help='directory of DPQA code', type=str)
parser.add_argument('--suffix', type=str,
                    help='suffix to the file name.')
args = parser.parse_args()

T_COHERENCE = 1500000  # unit is us

depth_json = {}
runtime_json = {}
duration_json = {}
# total duration (move, activate, deactivate) as percentage of coherence time

for size in [10, 12, 14, 16, 18, 20, 22, 30, 40, 50, 60, 70, 80, 90]:
    runtime_this_size = {}
    duration_this_size = {}
    depth_this_size = {}
    for id in range(10):

        # find the file containing SMT output.
        # skip the combinations of (size, id) that are missing
        dir = args.dir if args.dir else './results/smt/'
        path = dir + f"rand3reg_{str(size)}_{str(id)}"
        if args.suffix:
            path += '_' + args.suffix + '.json'
        else:
            path += '.json'
        if not os.path.isfile(path):
            continue

        with open(path, 'r') as f:
            data = json.load(f)
        runtime_this_size[str(id)] = data['duration']
        transfer = False if data['no_transfer'] else True

        CodeGen(path, no_transfer=data['no_transfer'], dir=args.out_dir)
        with open(
                f"./results/code/rand3reg_{str(size)}_{str(id)}" +
            ('_' + args.suffix if args.suffix else '') +
                "_code_full.json", 'r') as file:
            data = json.load(file)

        duration = 0
        depth = 0
        for inst in data:
            if inst['type'] == 'Move':
                if transfer or (not inst['name'].startswith('Reload')):
                    # if compiled with the hybrid approach (transfer=True)
                    # or, only count the big moves in the optimal results
                    duration += inst['duration']
            elif inst['type'] == 'Rydberg':
                duration += 0.15  # duration is 0.15us per Bluvstein et al.
                depth += 1
            elif inst['type'] == 'Deactivate' or inst['type'] == 'Activate':
                if transfer:
                    duration += 50  # per Ebadi et al.

        duration_this_size[str(id)] = duration / T_COHERENCE
        depth_this_size[str(id)] = depth
    runtime_json[str(size)] = runtime_this_size
    duration_json[str(size)] = duration_this_size
    depth_json[str(size)] = depth_this_size

with open("results/stats/dpqa_runtime.json", 'w') as file:
    json.dump(runtime_json, file)

with open("results/stats/dpqa_duration.json", 'w') as file:
    json.dump(duration_json, file)

with open("results/stats/dpqa_depth.json", 'w') as file:
    json.dump(depth_json, file)
