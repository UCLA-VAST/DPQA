from animation import CodeGen
import json

test_qubit = [10, 12, 14, 16, 18, 30, 40, 50, 60, 70, 80]
for i in test_qubit:
    for j in range(10):
        filename = "results/smt/rand3reg_{}_{}.json".format(i,j)
        print("test file {}".format(filename))
        codegen = CodeGen(filename, no_transfer=False, dir="test/codeGen/")