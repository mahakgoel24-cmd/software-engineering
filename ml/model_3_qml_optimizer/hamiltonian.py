# ml/model_3_qml_optimizer/hamiltonian.py

import numpy as np
from qiskit.quantum_info import SparsePauliOp

def build_ising_hamiltonian(scores, salaries, hours, availability, k):
    """
    Build Ising Hamiltonian for candidate selection
    """

    n = len(scores)
    h = np.zeros(n)
    J = {}

    # Linear terms (quality vs cost)
    for i in range(n):
        h[i] = (
            -2.0 * scores[i]
            + 0.5 * salaries[i]
            + 0.3 * hours[i]
        )

        # Availability penalty
        if hours[i] > availability[i]:
            h[i] += 2.0 * (hours[i] - availability[i])

    # Constraint: select exactly k
    for i in range(n):
        for j in range(i + 1, n):
            J[(i, j)] = 2.0

        h[i] += -2.0 * k

    return h, J
