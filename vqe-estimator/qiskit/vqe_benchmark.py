"""
Variational Quantum Eigensolver Benchmark Program - Qiskit
"""

import json
import os
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter

sys.path[1:1] = ["_common", "_common/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit"]
import execute as ex
import metrics as metrics

# Benchmark Name
benchmark_name = "VQE Simulation"

verbose = False

# saved circuits for display
QC_ = None
Hf_ = None
CO_ = None

################### Circuit Definition #######################################


# Construct a Qiskit circuit for VQE Energy evaluation with UCCSD ansatz
# param: n_spin_orbs - The number of spin orbitals.
# return: return a Qiskit pubs for this VQE ansatz
def VQEEnergy(n_spin_orbs, na, nb, circuit_id=0, method=1):
    # number of alpha spin orbitals
    norb_a = int(n_spin_orbs / 2)

    # construct the Hamiltonian
    qubit_op = ReadHamiltonian(n_spin_orbs)

    # allocate qubits
    num_qubits = n_spin_orbs

    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr, name=f"vqe-ansatz({method}) {num_qubits} {circuit_id}")

    # initialize the HF state
    Hf = HartreeFock(num_qubits, na, nb)
    qc.append(Hf, qr)

    # form the list of single and double excitations
    excitationList = []
    for occ_a in range(na):
        for vir_a in range(na, norb_a):
            excitationList.append((occ_a, vir_a))

    for occ_b in range(norb_a, norb_a + nb):
        for vir_b in range(norb_a + nb, n_spin_orbs):
            excitationList.append((occ_b, vir_b))

    for occ_a in range(na):
        for vir_a in range(na, norb_a):
            for occ_b in range(norb_a, norb_a + nb):
                for vir_b in range(norb_a + nb, n_spin_orbs):
                    excitationList.append((occ_a, vir_a, occ_b, vir_b))

    # get cluster operators in Paulis
    pauli_list = readPauliExcitation(n_spin_orbs, circuit_id)

    # loop over the Pauli operators
    for index, PauliOp in enumerate(pauli_list):
        # get circuit for exp(-iP)
        cluster_qc = ClusterOperatorCircuit(PauliOp, excitationList[index])

        # add to ansatz
        qc.append(cluster_qc, [i for i in range(cluster_qc.num_qubits)])

    # save circuit
    global QC_
    if QC_ is None:
        if qc.num_qubits < 7:
            QC_ = qc

    # method 1, only compute the last term in the Hamiltonian
    if method == 1:
        # last term in Hamiltonian
        op = qubit_op[1]
        qc.metadata["method1"] = str(op.paulis[0]) + " " + str(np.real(op.coeffs)[0])
        return qc, qubit_op[1]

    global normalization
    # ignore the identity matrix qubit_op[0]
    qubit_op = qubit_op[1:]
    normalization = sum(abs(p.coeffs[0]) for p in qubit_op)
    normalization /= len(qubit_op.group_commuting(qubit_wise=True))
    qc.metadata["method2"] = qubit_op
    return qc, list(qubit_op)


# Function that constructs the circuit for a given cluster operator
def ClusterOperatorCircuit(pauli_op, excitationIndex):
    num_qubits = pauli_op.num_qubits

    # compute exp(-iP) with 1st order Trotter step
    qc_op = PauliEvolutionGate(pauli_op, synthesis=LieTrotter())
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_op, range(num_qubits))
    qc.name = f"Cluster Op {excitationIndex}"

    global CO_
    if CO_ == None or qc.num_qubits <= 4:
        if qc.num_qubits < 7:
            CO_ = qc

    # return this circuit
    return qc


# Function that adds expectation measurements to the raw circuits
def ExpectationCircuit(qc, pauli, nqubit, method=2):
    # copy the unrotated circuit
    raw_qc = qc.copy()

    # whether this term is diagonal
    is_diag = True

    # primitive Pauli string
    PauliString = pauli.to_list()[0][0]

    # coefficient
    coeff = pauli.coeffs[0]

    # basis rotation
    for i, p in enumerate(PauliString):
        target_qubit = nqubit - i - 1
        if p == "X":
            is_diag = False
            raw_qc.h(target_qubit)
        elif p == "Y":
            raw_qc.sdg(target_qubit)
            raw_qc.h(target_qubit)
            is_diag = False

    # perform measurements
    raw_qc.measure_all()

    # name of this circuit
    raw_qc.name = PauliString + " " + str(np.real(coeff))

    # save circuit
    global QC_
    if QC_ == None or nqubit <= 4:
        if nqubit < 7:
            QC_ = raw_qc

    return raw_qc, is_diag


# Function that implements the Hartree-Fock state
def HartreeFock(norb, na, nb):
    # initialize the quantum circuit
    qc = QuantumCircuit(norb, name="Hf")

    # alpha electrons
    for ia in range(na):
        qc.x(ia)

    # beta electrons
    for ib in range(nb):
        qc.x(ib + int(norb / 2))

    # Save smaller circuit
    global Hf_
    if Hf_ == None or norb <= 4:
        if norb < 7:
            Hf_ = qc

    # return the circuit
    return qc


################ Helper Functions


# Function that converts a list of single and double excitation operators to Pauli operators
def readPauliExcitation(norb, circuit_id=0):
    # load pre-computed data
    filename = os.path.join(
        os.path.dirname(__file__), f"./ansatzes/{norb}_qubit_{circuit_id}.txt"
    )
    with open(filename) as f:
        data = f.read()
    ansatz_dict = json.loads(data)

    # initialize Pauli list
    pauli_list = []

    # current coefficients
    cur_coeff = 1e5

    # current Pauli list
    cur_list = []

    # loop over excitations
    for ext in ansatz_dict:
        if cur_coeff > 1e4:
            cur_coeff = ansatz_dict[ext]
            cur_list = [(ext, ansatz_dict[ext])]
        elif abs(abs(ansatz_dict[ext]) - abs(cur_coeff)) > 1e-4:
            pauli_list.append(SparsePauliOp.from_list(cur_list))
            cur_coeff = ansatz_dict[ext]
            cur_list = [(ext, ansatz_dict[ext])]
        else:
            cur_list.append((ext, ansatz_dict[ext]))

    # add the last term
    pauli_list.append(SparsePauliOp.from_list(cur_list))

    # return Pauli list
    return pauli_list


# Get the Hamiltonian by reading in pre-computed file
def ReadHamiltonian(nqubit):
    # load pre-computed data
    filename = os.path.join(
        os.path.dirname(__file__), f"./Hamiltonians/{nqubit}_qubit.txt"
    )
    with open(filename) as f:
        data = f.read()
    ham_dict = json.loads(data)

    # pauli list
    pauli_list = []
    for p in ham_dict:
        pauli_list.append((p, ham_dict[p]))

    # build Hamiltonian
    ham = SparsePauliOp.from_list(pauli_list)

    # return Hamiltonian
    return ham


################ Result Data Analysis


## Analyze and print measured results
## Compute the quality of the result based on measured probability distribution for each state
def analyze_and_print_result(qc, result, num_qubits, references, _num_shots):
    method = 0
    if "method1" in qc.metadata:
        method = 1
        # total circuit name (pauli string + coefficient) for method 1
        total_name = qc.metadata["method1"]
        circuit_id = int(total_name.split()[2])
        ref = references[f"Qubits - {num_qubits} - {circuit_id}"]
    elif "method2" in qc.metadata:
        method = 2
        qubit_op: SparsePauliOp = qc.metadata["method2"]
    else:
        raise RuntimeError("Either method 1 or 2 should be chosen")

    expval = result.get_expectation_values(qc)
    if method == 1:
        fidelity = metrics.accuracy_ratio_fidelity(
            expval, ref["exact"], ref["min"], ref["max"]
        )
        if verbose:
            print(f"... fidelity = {fidelity}")
        return fidelity

    # modify fidelity based on the coefficient (only for method 2)
    total_fidelity = {"fidelity": 0}
    for val, (pauli, _) in zip(expval, qubit_op.to_list()):
        ref = references[pauli]
        fidelity = metrics.accuracy_ratio_fidelity(
            val, ref["exact"], ref["min"], ref["max"]
        )
        total_fidelity["fidelity"] += fidelity["fidelity"]
        if verbose:
            print(f"... {pauli=} {fidelity=}")
    total_fidelity["fidelity"] /= qubit_op.size

    if verbose:
        print(f"... {total_fidelity=}")

    return total_fidelity


################ Benchmark Loop

# Max qubits must be 12 since the referenced files only go to 12 qubits
MAX_QUBITS = 12


# Execute program with default parameters
def run(
    min_qubits=4,
    max_qubits=8,
    skip_qubits=1,
    max_circuits=3,
    num_shots=4092,
    method=1,
    backend_id=None,
    provider_backend=None,
    hub="ibm-q",
    group="open",
    project="main",
    exec_options=None,
    context=None,
):
    print(f"{benchmark_name} ({method}) Benchmark Program - Qiskit")

    max_qubits = max(max_qubits, min_qubits)  # max must be >= min

    # validate parameters (smallest circuit is 4 qubits and largest is 10 qubitts)
    max_qubits = min(max_qubits, MAX_QUBITS)
    min_qubits = min(max(4, min_qubits), max_qubits)
    if min_qubits % 2 == 1:
        min_qubits += 1  # min_qubits must be even
    skip_qubits = max(1, skip_qubits)

    if method == 2:
        max_circuits = 1

    if max_qubits < 4:
        print(
            f"Max number of qubits {max_qubits} is too low to run method {method} of VQE algorithm"
        )
        return

    if backend_id == "statevector_estimator":
        precision = 0
    else:
        precision = 1.0 / np.sqrt(num_shots)

    # create context identifier
    if context is None:
        context = f"{benchmark_name} ({method}) Benchmark"

    ##########

    # Initialize the metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler(qc, result, num_qubits, type, num_shots):
        # load pre-computed data
        if "method1" in qc.metadata:
            filename = os.path.join(
                os.path.dirname(__file__),
                f"../_common/precalculated_data_{num_qubits}_qubit_method1.json",
            )
            with open(filename) as f:
                references = json.load(f)
        elif "method2" in qc.metadata:
            filename = os.path.join(
                os.path.dirname(__file__),
                f"../_common/precalculated_data_{num_qubits}_qubit_method2.json",
            )
            with open(filename) as f:
                references = json.load(f)
        else:
            raise RuntimeError("Either method 1 or 2 should be chosen")

        fidelity = analyze_and_print_result(
            qc, result, num_qubits, references, num_shots
        )

        if "method1" in qc.metadata:
            circuit_id = qc.metadata["method1"].split()[2]
            metrics.store_metric(num_qubits, circuit_id, "fidelity", fidelity)
        elif "method2" in qc.metadata:
            metrics.store_metric(num_qubits, "method2", "fidelity", fidelity)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(
        backend_id,
        provider_backend=provider_backend,
        hub=hub,
        group=group,
        project=project,
        exec_options=exec_options,
        context=context,
    )

    ##########

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for input_size in range(min_qubits, max_qubits + 1, 2):
        # reset random seed
        np.random.seed(0)

        # determine the number of circuits to execute for this group
        num_circuits = min(3, max_circuits)

        num_qubits = input_size

        # decides number of electrons
        na = int(num_qubits / 4)
        nb = int(num_qubits / 4)

        # random seed
        np.random.seed(0)

        # create the circuit for given qubit size and simulation parameters, store time metric
        ts = time.time()

        # pub list
        pub_list = []

        # Method 1 (default)
        if method == 1:
            # loop over circuits
            for circuit_id in range(num_circuits):
                # construct circuit and observable
                pub = VQEEnergy(num_qubits, na, nb, circuit_id, method)
                pub[0].metadata["method1"] += " " + str(circuit_id)

                # add to list
                pub_list.append(pub)
        # method 2
        elif method == 2:
            # construct circuit and all observables
            pub = VQEEnergy(num_qubits, na, nb, 0, method)
            pub_list.append(pub)

        print(
            f"************\nExecuting [{len(pub_list)}] pubs with num_qubits = {num_qubits}"
        )

        for pub in pub_list:
            qc = pub[0]

            # get circuit id
            if "method1" in qc.metadata:
                circuit_id = qc.metadata["method1"].split()[2]
            else:
                circuit_id = "method2"

            # record creation time
            metrics.store_metric(
                input_size, circuit_id, "create_time", time.time() - ts
            )

            # collapse the sub-circuits used in this benchmark (for qiskit)
            pub2 = EstimatorPub.coerce((qc, pub[1]))

            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_pub(pub2, input_size, circuit_id, precision)

        # Wait for some active circuits to complete; report metrics when group complete
        ex.throttle_execution(metrics.finalize_group)

    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    ##########

    # print a sample circuit
    print("Sample Circuit:")
    print(QC_ if QC_ != None else "  ... too large!")
    print("\nHartree Fock Generator 'Hf' =")
    print(Hf_ if Hf_ != None else " ... too large!")
    print("\nCluster Operator Example 'Cluster Op' =")
    print(CO_ if CO_ != None else " ... too large!")

    # Plot metrics for all circuit sizes
    metrics.plot_metrics(f"Benchmark Results - {benchmark_name} ({method}) - Qiskit")


#######################
# MAIN

import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Variational Quantum Eigensolver Benchmark"
    )
    # parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    # parser.add_argument("--target", "-t", default=None, help="Target Backend", type=str)
    parser.add_argument(
        "--backend_id", "-b", default=None, help="Backend Identifier", type=str
    )
    parser.add_argument(
        "--num_qubits",
        "-n",
        default=0,
        help="Number of qubits (min = max = N)",
        type=int,
    )
    parser.add_argument(
        "--min_qubits", "-min", default=4, help="Minimum number of qubits", type=int
    )
    parser.add_argument(
        "--max_qubits", "-max", default=8, help="Maximum number of qubits", type=int
    )
    parser.add_argument(
        "--skip_qubits", "-k", default=1, help="Number of qubits to skip", type=int
    )
    parser.add_argument(
        "--max_circuits", "-c", default=3, help="Maximum circuit repetitions", type=int
    )
    parser.add_argument(
        "--num_shots", "-s", default=4092, help="Number of shots", type=int
    )
    parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
    parser.add_argument(
        "--nonoise", "-non", action="store_true", help="Use Noiseless Simulator"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")

    return parser.parse_args()


if __name__ == "__main__":
    import argparse

    args = get_args()

    # special argument handling
    ex.verbose = args.verbose
    verbose = args.verbose

    if args.num_qubits > 0:
        args.min_qubits = args.max_qubits = args.num_qubits

    # execute benchmark program
    run(
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits,
        max_circuits=args.max_circuits,
        num_shots=args.num_shots,
        method=args.method,
        backend_id=args.backend_id,
        exec_options={"noise_model": None} if args.nonoise else {},
        # api=args.api
    )
