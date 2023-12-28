from solve import DPQA
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('file', metavar='F', type=str, help='qasm file')
parser.add_argument('--suffix', type=str,
                    help='suffix to the file name.')
parser.add_argument('--dir', help='output directory', type=str)
parser.add_argument('--print_detail', action='store_true')
args = parser.parse_args()

filename = args.file.split('/')[-1].split('.')[0] 
if args.suffix:
    filename += '_' + args.suffix
tmp = DPQA(
    filename,
    dir=args.dir if args.dir else './results/smt/',
    print_detail=args.print_detail
)
tmp.setArchitecture([16, 16, 16, 16])
with open('only_2q.json', 'r') as f:
    only_2q = json.load(f)
tmp.setProgram(only_2q[args.file])
tmp.setOptimalRatio(0)
result = tmp.solve(save_file=True)
depth = 0
num_2q = 0
for layer in result['layers']:
    num_2q += len(layer['gates'])
    if len(layer['gates']) != 0:
        depth += 1
with open('olsq_2q', 'a') as f:
    f.write(f'{filename}, {num_2q}, {depth}, {result["duration"]}\n')
