from solve import DPQA
import argparse
import json

with open('./graphs.json', 'r') as f:
    graphs = json.load(f)
parser = argparse.ArgumentParser()
parser.add_argument('size', metavar='S', type=int, help='#qubit in graph.')
parser.add_argument('id', metavar='I', type=int, help='index of the graph.')
parser.add_argument('--suffix', type=str,
                    help='suffix to the file name.')
parser.add_argument('--dir', help='output directory', type=str)
parser.add_argument('--print_detail', action='store_true')
args = parser.parse_args()

filename = 'rand3reg_' + str(args.size) + '_' + str(args.id)
if args.suffix:
    filename += '_' + args.suffix
tmp = DPQA(
    filename,
    dir=args.dir if args.dir else './results/smt/',
    print_detail=args.print_detail
)
tmp.setArchitecture([16, 16, 16, 16])
if str(args.size) in graphs.keys() and args.id in range(10):
    tmp.setProgram(graphs[str(args.size)][args.id])
else:
    raise ValueError(f'No such graph {args.size}_{args.id}.')
tmp.setCommutation()
tmp.hybrid_strategy()
tmp.solve(save_file=True)
