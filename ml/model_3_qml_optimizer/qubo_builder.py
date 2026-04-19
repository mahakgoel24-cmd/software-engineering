def build_qubo(scores, salaries, hours, availability, k):
    """
    Build Ising/QUBO coefficients for candidate selection.

    Args:
        scores (list[float]): normalized ranking scores from Model 2
        salaries (list[float]): expected salary per candidate
        hours (list[float]): estimated project hours
        availability (list[float]): hours per week available
        k (int): number of candidates to select

    Returns:
        h (list[float]): linear Ising coefficients
        J (dict[(int,int), float]): quadratic Ising coefficients
    """

    n = len(scores)

    # ---- Normalize helpers ----
    def norm(x):
        m, M = min(x), max(x)
        return [(v - m) / (M - m + 1e-6) for v in x]

    scores_n = norm(scores)
    salaries_n = norm(salaries)
    hours_n = norm(hours)
    avail_n = norm(availability)

    # ---- Linear terms (h_i) ----
    h = []
    for i in range(n):
        energy = (
            -1.5 * scores_n[i]       # reward good candidates
            + 1.0 * salaries_n[i]    # penalize high salary
            + 0.7 * hours_n[i]       # penalize long timelines
            - 0.8 * avail_n[i]       # reward availability
        )
        h.append(energy)

    # ---- Quadratic constraint: select exactly k ----
    # (sum x_i - k)^2 = sum x_i^2 + 2 sum x_i x_j - 2k sum x_i + k^2
    # x_i^2 = x_i for binary variables

    penalty = 2.0
    J = {}

    for i in range(n):
        h[i] += penalty * (1 - 2 * k)

        for j in range(i + 1, n):
            J[(i, j)] = J.get((i, j), 0) + 2 * penalty

    return h, J
