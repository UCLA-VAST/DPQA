# OLSQ-DPQA Compiler
Optimal Layout Synthesizer of Quantum Circuits for Dynamically Field-Programmable Qubits Array.
Open source under the BSD 3-Clause license.

Repo structure:
- `run.py` is an example of using the compiler. Refer to `python run.py -h` for options.
- `solve.py` contains the class `DPQA` where we encode the compilation problem to SMT, and use `z3-solver` to solve it.
- `graphs.json` contains some random 3-regular graphs.
- `animation.py` contains the class `CodeGen` that generates DPQA instructions (five types `Init`, `Rydberg`, `Activate`, `Deactivate`, and `Move`), and the class `Animator` that generates animations from DPQA instructions. Refer to `python animation.py -h` for options.
- `results/` is the default directory for the results.
  - `results/smt/` contains the output of SMT variable assignments.
  - `results/code/` contains the code files generated from SMT output.
  - `results/animations/` contains a few example animations generated from code files.

How to use the compiler:
- We used a Python 3 environment with `z3-solver`, `networkx`, and `python-sat`, and `matplotlib`. The Python scripts are run in the root directory of the repo.
- Run `python run.py <S> <I>` where `<S>` is the size of the random 3-regular graph, `<I>` is the id of the graph. To try other graphs, please edit `run.py` as needed.
- The specific runtimes can differ because of the hardware, environment, and updates of this repo. Please refer to branch(es) of this repo for specific versions corresponding to the paper(s), e.g., [2306.03487rev1](https://github.com/UCLA-VAST/DPQA/tree/2306.03487rev1)
- (Optional) To generate animation, run `python animation.py <F>` where `<F>` is the SMT output file, e.g., `results/smt/rand3reg_90_4.json`.

Explaination of `run.py`:
- The main class is named `DPQA` which is in `solve.py`. 
  - When we initialize it, there is a mandatory argument `name` which is used for saving output file (a JSON containing SMT variables).
  - Optionally, you can specify the directory for this file with argument `dir`.
  - There is another optional argument `print_detail` to specify the granularity of printout.
- We need to specify the architecture with `setArchitecture` method of `DPQA`. It takes in a list of 4 numbers, which are the number of columns of interaction sites, the number of rows of interaction sites, the number of AOD columns, and the number of AOD rows.
- We need to specify the two-qubit gates in a list with `setProgram` method of `DPQA`. For example, a circuit CZ(0,1), CZ(1,2), CZ(0,1) will be `[[0,1], [1,2], [0,1]]`.
- If all the gates are commutable with each other (e.g., the two-qubit gates in an iteration of QAOA), call the `setCommutation` method of `DPQA`. Otherwise, do not call it and the compiler will process the gates considering dependency.
- We can set the ratio of switching from interative peeling to optimal (multi-stage) solving with `setOptimalRatio` method of `DPQA`. By default, the ratio is 0.
- Finally, we can solve the formulated SMT problem with the `solve` method of `DPQA`.
