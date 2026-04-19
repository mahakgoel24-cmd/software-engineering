# ml/model_3_qml_optimizer/qaoa_solver.py

from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler
from qiskit.quantum_info import SparsePauliOp


def solve_with_qaoa(h, J, reps=2):
    n = len(h)

    paulis = []
    coeffs = []

    for i in range(n):
        z = ["I"] * n
        z[i] = "Z"
        paulis.append("".join(z))
        coeffs.append(h[i])

    for (i, j), val in J.items():
        z = ["I"] * n
        z[i] = "Z"
        z[j] = "Z"
        paulis.append("".join(z))
        coeffs.append(val)

    hamiltonian = SparsePauliOp(paulis, coeffs)

    qaoa = QAOA(
        sampler=Sampler(),
        optimizer=COBYLA(maxiter=100),
        reps=reps
    )

    result = qaoa.compute_minimum_eigenvalue(hamiltonian)

    probs = result.eigenstate.binary_probabilities()
    best = max(probs, key=probs.get)

    return [int(b) for b in best[::-1]]
