import json
from argparse import ArgumentParser
from os import path

import numpy as np
from qiskit import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2

from vqe_benchmark import VQEEnergy

NUM_CIRCUITS = 3


def method1(num_qubits: int) -> dict:
    na = nb = num_qubits // 4
    method = 1
    pm = generate_preset_pass_manager(optimization_level=2)

    # generate pubs
    pubs = []
    for circuit_id in range(NUM_CIRCUITS):
        pub = VQEEnergy(num_qubits, na, nb, circuit_id, method)
        circuit = pm.run(pub[0])
        pub = (circuit, pub[1])
        pubs.append(pub)

    # compute the exact expectation values
    data = {}
    estimator = EstimatorV2()
    # estimator = StatevectorEstimator()
    result = estimator.run(pubs).result()
    for circuit_id, pub in enumerate(pubs):
        exact = result[circuit_id].data.evs.item()
        min_eigval, max_eigval = eigvals(pub[1])
        data[f"Qubits - {num_qubits} - {circuit_id}"] = {
            "exact": exact,
            "min": min_eigval,
            "max": max_eigval,
        }
    return data


def method2(num_qubits: int) -> None:
    na = nb = num_qubits // 4
    method = 2
    pm = generate_preset_pass_manager(optimization_level=2)

    # generate a pub
    pub = VQEEnergy(num_qubits, na, nb, 0, method)
    circuit = pm.run(pub[0])
    ops = pub[1]
    pub = (circuit, pub[1])

    # compute the exact expectation values
    data = {}
    estimator = EstimatorV2()
    # estimator = StatevectorEstimator()
    result = estimator.run([pub]).result()
    exact = result[0].data.evs
    for index, op in enumerate(ops):
        min_eigval, max_eigval = eigvals(op)
        data[f"{op.paulis[0]}"] = {
            "exact": exact[index],
            "min": min_eigval,
            "max": max_eigval,
        }
    return data


def eigvals(op: SparsePauliOp) -> tuple[float, float]:
    eigvals = np.linalg.eigvalsh(op.to_matrix())
    min_eigval = min(eigvals).item()
    max_eigval = max(eigvals).item()
    return min_eigval, max_eigval


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--output", "-o", help="Output directory name", type=str, default="../_common"
    )
    parser.add_argument("--method", "-m", help="Method type", type=int, default=1)
    args = parser.parse_args()
    if not args.output:
        parser.print_usage()
    return args


def save_file(data: dict, filename: str) -> None:
    with open(filename, "w") as file:
        json.dump(data, file, indent=4, sort_keys=True)


def main():
    args = get_args()
    print(args)
    if not args.output:
        return
    for num_qubits in range(4, 13, 2):
        print(num_qubits)
        if args.method == 1:
            data = method1(num_qubits)
        elif args.method == 2:
            data = method2(num_qubits)
        else:
            raise ValueError(f"Invalid method type ({args.method}). Should be 1 or 2.")
        filename = path.join(
            args.output,
            f"precalculated_data_{num_qubits}_qubit_method{args.method}.json",
        )
        save_file(data, filename)


if __name__ == "__main__":
    main()
