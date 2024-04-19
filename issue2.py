from solve import DPQA
from animation import CodeGen
import json

dpqa = DPQA("my_graph", dir="./results/smt/")
dpqa.setArchitecture([16, 16, 16, 16])
graph = [[5, 6], [5, 6], [4, 6], [3, 6], [2, 6], [1, 6], [4, 6], [3, 6], [2, 6],
         [1, 6], [4, 5], [3, 5], [2, 5], [1, 5], [0, 6], [4, 5], [3, 5], [2, 5],
         [1, 5], [0, 6], [3, 4], [2, 4], [1, 4], [0, 5], [3, 4], [2, 4], [1, 4],
         [0, 5], [2, 3], [1, 3], [0, 4], [2, 3], [1, 3], [0, 4], [1, 2], [0, 3],
         [2, 4], [1, 2], [0, 3], [2, 4], [0, 2], [2, 4], [0, 2], [0, 1], [1, 5],
         [0, 1], [1, 5], [1, 5], [0, 6], [0, 6], [0, 6]]
dpqa.setProgram(graph)
dpqa.hybrid_strategy()
dpqa.solve(save_file=True)

with open("./results/smt/my_graph.json") as f:
    data = json.load(f)
codegen = CodeGen("./results/smt/my_graph.json", no_transfer=data['no_transfer'], dir="./results/code/")