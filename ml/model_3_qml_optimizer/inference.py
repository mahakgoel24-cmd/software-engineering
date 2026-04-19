from .qubo_builder import build_qubo
from .qaoa_solver import solve_with_qaoa


def quantum_select_top_10_percent(
    ranked_user_ids,
    scores,
    salaries,
    hours,
    availability
):
    n = len(ranked_user_ids)
    k = max(1, int(0.10 * n))

    h, J = build_qubo(
        scores=scores,
        salaries=salaries,
        hours=hours,
        availability=availability,
        k=k
    )

    solution = solve_with_qaoa(h, J)

    selected_indices = [i for i, b in enumerate(solution) if b == 1]
    selected_user_ids = [ranked_user_ids[i] for i in selected_indices[:k]]

    return selected_user_ids
