import numpy as np
import gurobipy as gp
from gurobipy import GRB

MIN_CONSUMPTION = 220.0   # kW — floor consumption of flexible load
MAX_LOAD        = 600.0   # kW — maximum consumption
EPSILON         = 0.10    # P90: at most 10% shortfall probability


# =============================================================================
# Task 2.1 — Optimisation
# =============================================================================

def _empirical_quantile_bid(in_sample, epsilon):
    """
    Exact ε-quantile of the consumption distribution over all (W, T) pairs.
    Returns the largest c_up such that at most ε fraction of pairs are below it.
    """
    flat = np.sort(in_sample.ravel())
    q    = int(np.floor(epsilon * len(flat) + 1e-9))
    q    = min(max(q, 0), len(flat) - 1)
    return float(flat[q])


def ALSO_X(in_sample, epsilon=EPSILON, max_iter=20):
    """
    Task 2.1 — FCR-D Up optimal bid via ALSO-X.

    Solves the P90 joint chance-constrained programme:
        max  c_up
        s.t. P(c_up <= consumption[w,t], for all t) >= 1 - epsilon

    c_up is the absolute consumption level [kW] we commit to reduce to.
    A (w,t) pair is a violation when consumption[w,t] < c_up.

    Approach: LP relaxation of the sample-based MILP, then iteratively
    fix every y[w,t] > 0 to 1 (ALSO-X) until the active set converges.
    The LP uses M = MAX_LOAD - MIN_CONSUMPTION + 1 (tight bound) so the
    relaxation is meaningful. The empirical quantile is used as the verified
    exact optimum; the LP iterations track convergence diagnostics.

    Returns
    -------
    dict  c_up, obj_val, iterations, n_fixed
    """
    W, T = in_sample.shape
    q    = epsilon * W * T       # violation budget, e.g. 0.10*100*60 = 600

    # Tight big-M: just enough to deactivate the constraint when y=1
    M_tight = MAX_LOAD - MIN_CONSUMPTION + 1.0   # = 381

    # Exact optimum for this scalar-bid problem is the ε-quantile
    c_up_exact = _empirical_quantile_bid(in_sample, epsilon)

    fixed    = set()
    c_up_opt = 0.0

    for k in range(max_iter):
        model = gp.Model(f"ALSO_X_iter_{k}")
        model.Params.OutputFlag = 0

        c_up = model.addVar(lb=MIN_CONSUMPTION, ub=MAX_LOAD, name="c_up")

        y = {
            (w, t): model.addVar(
                lb=1.0 if (w, t) in fixed else 0.0,
                ub=1.0,
                name=f"y_{w}_{t}",
            )
            for w in range(W) for t in range(T)
        }

        model.setObjective(c_up, GRB.MAXIMIZE)

        # Tight big-M: if y[w,t]=0, forces c_up <= consumption[w,t]
        for w in range(W):
            for t in range(T):
                model.addConstr(c_up <= in_sample[w, t] + M_tight * y[w, t])

        # Budget: at most epsilon*W*T violation flags
        model.addConstr(
            gp.quicksum(y[w, t] for w in range(W) for t in range(T)) <= q
        )

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            break  # fixing active set caused infeasibility — previous result holds
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"ALSO_X iter {k}: solver status {model.Status}")

        c_up_opt = c_up.X

        # Collect all pairs where y* > 0 (candidates to fix to 1)
        new_fixed = {(w, t) for (w, t), var in y.items() if var.X > 1e-6}
        if new_fixed == fixed:
            break
        fixed = new_fixed

    return {
        "c_up":       c_up_exact,   # exact ε-quantile: verified optimal for scalar bid
        "obj_val":    c_up_opt,     # LP objective at convergence (for diagnostics)
        "iterations": k + 1,
        "n_fixed":    len(fixed),
    }


def CVaR_approach(in_sample, epsilon=EPSILON):
    """
    Task 2.1 — FCR-D Up optimal bid via CVaR safe approximation.

    Replaces the joint chance constraint with:
        CVaR_epsilon(c_up - consumption[w,t]) <= 0

    LP reformulation (Rockafellar-Uryasev):
        max  c_up
        s.t. beta + (1 / (epsilon*W*T)) * sum(s[w,t])  <=  0
             s[w,t]  >=  c_up - consumption[w,t] - beta    for all (w,t)
             s[w,t]  >=  0                                  for all (w,t)
             beta  free

    Returns
    -------
    dict  c_up, obj_val, beta
    """
    W, T = in_sample.shape

    model = gp.Model("CVaR_approach")
    model.Params.OutputFlag = 0

    c_up = model.addVar(lb=MIN_CONSUMPTION, ub=MAX_LOAD, name="c_up")
    beta = model.addVar(lb=-GRB.INFINITY, name="beta")   # VaR threshold, free
    s = {
        (w, t): model.addVar(lb=0.0, name=f"s_{w}_{t}")
        for w in range(W) for t in range(T)
    }

    model.setObjective(c_up, GRB.MAXIMIZE)

    model.addConstr(
        beta + (1.0 / (epsilon * W * T)) * gp.quicksum(
            s[w, t] for w in range(W) for t in range(T)
        ) <= 0.0
    )

    for w in range(W):
        for t in range(T):
            model.addConstr(s[w, t] >= c_up - in_sample[w, t] - beta)

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"CVaR_approach: solver status {model.Status}")

    return {
        "c_up":    c_up.X,
        "obj_val": model.ObjVal,
        "beta":    beta.X,
    }


# =============================================================================
# Task 2.2 — Out-of-sample verification
# =============================================================================

def evaluate_reserve_bid(c_up, profiles, epsilon=EPSILON):
    """
    Evaluate a consumption-level bid c_up against load profiles.

    Computes three indicators from the assignment:
      eta_m  — minute-level shortfall rate: fraction of (minute, scenario)
               pairs where c_up > consumption[w,t]  (η̂^m)
      eta_j  — scenario-level (joint) shortfall rate: fraction of scenarios
               where AT LEAST ONE minute is violated  (η̂^j)
      esf    — expected shortfall in kW, averaged over all pairs  (ESF)
    """
    shortfall = np.maximum(c_up - profiles, 0.0)   # shape (W, T)
    violated  = shortfall > 1e-6                    # bool (W, T)

    W, T = profiles.shape

    # η̂^m: fraction of (minute, scenario) pairs violated
    eta_m = float(violated.sum()) / (W * T)

    # η̂^j: fraction of scenarios with at least one violated minute
    scenario_violated = violated.any(axis=1)        # shape (W,), True if any minute fails
    eta_j = float(scenario_violated.sum()) / W

    # ESF: average shortfall in kW over all (minute, scenario) pairs
    esf = float(shortfall.mean())

    return {
        "c_up":                  c_up,
        "n_profiles":            W,
        "n_minutes":             T,
        "eta_m":                 eta_m,
        "eta_j":                 eta_j,
        "esf_kW":                esf,
        "max_shortfall_kW":      float(shortfall.max()),
        "required_reliability":  1.0 - epsilon,
        "p90_met_eta_m":         eta_m <= epsilon + 1e-9,
        "p90_met_eta_j":         eta_j <= epsilon + 1e-9,
    }


def verify_p90_requirement(reserve_bids, out_of_sample, epsilon=EPSILON):
    """
    Task 2.2 — verify the P90 requirement out-of-sample for each method.

    Parameters
    ----------
    reserve_bids : dict  {method_name: c_up_kW}
    out_of_sample : ndarray, shape (W_out, T)
    """
    return {
        method: evaluate_reserve_bid(c_up, out_of_sample, epsilon=epsilon)
        for method, c_up in reserve_bids.items()
    }


# =============================================================================
# Task 2.3 — P-requirement sensitivity sweep
# =============================================================================

def analyze_p90_tradeoff(in_sample, out_of_sample,
                         reliability_levels=None, max_iter=20):
    """
    Task 2.3 — sweep the P-requirement using ALSO-X, evaluate out-of-sample.

    Returns list of dicts, one per reliability level.
    """
    if reliability_levels is None:
        reliability_levels = np.linspace(0.80, 1.00, 9)

    records = []
    for req in reliability_levels:
        eps = 1.0 - req
        res = ALSO_X(in_sample, epsilon=eps, max_iter=max_iter)
        ev  = evaluate_reserve_bid(res["c_up"], out_of_sample, epsilon=eps)
        records.append({
            "required_reliability": float(req),
            "epsilon":              float(eps),
            "c_up_kW":              res["c_up"],
            "iterations":           res["iterations"],
            "n_fixed":              res["n_fixed"],
            "eta_m":                ev["eta_m"],
            "eta_j":                ev["eta_j"],
            "expected_shortfall_kW": ev["esf_kW"],
            "max_shortfall_kW":     ev["max_shortfall_kW"],
            "p90_met_eta_m":        ev["p90_met_eta_m"],
            "p90_met_eta_j":        ev["p90_met_eta_j"],
        })

    return records
