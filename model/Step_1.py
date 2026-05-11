import random
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from scenario_builder import filter_scenarios


def evaluate_profit(q_DA, sc, hours):
    """Compute expected profit for a fixed q_DA offer evaluated on sc (no LP)."""
    total = 0.0
    for (w, p, i) in sc["scenarios"]:
        scenario_profit = 0.0
        for t in hours:
            imb = sc["p_wind"][(w, p, i, t)] - q_DA[t]
            dp_plus  = max(imb,  0.0)
            dp_minus = max(-imb, 0.0)
            scenario_profit += (
                sc["lda"][(w, p, i, t)]          * q_DA[t]
                + sc["lambda_up"][(w, p, i, t)]  * dp_plus
                - sc["lambda_down"][(w, p, i, t)] * dp_minus
            )
        total += sc["prob"][(w, p, i)] * scenario_profit
    return total


def Stochastic_offering_strategy(data, sc, scheme=None):
    """
    Task 1.1 — Stochastic offering strategy under the one-price scheme.

    Parameters
    ----------
    data : dict from input_data()
    sc   : dict from build_scenarios(data, scheme='one_price')

    Returns
    -------
    results : dict
        q_DA            : dict {t: MW}          optimal day-ahead offer per hour
        expected_profit : float                  expected total profit (€)
        profit_scenario : dict {(w,p,i): €}     total profit per combined scenario
        delta_p_plus    : dict {(w,p,i,t): MW}  surplus imbalance
        delta_p_minus   : dict {(w,p,i,t): MW}  deficit imbalance
        obj_val         : float                  LP objective value
        scheme          : str
    """
    hours     = data["hours"]
    P_max     = data["P_max"]
    scenarios = sc["scenarios"]
    prob      = sc["prob"]
    p_wind    = sc["p_wind"]
    lda       = sc["lda"]
    lambda_up   = sc["lambda_up"] 
    lambda_down   = sc["lambda_down"]

    model = gp.Model("Stochastic_offering_strategy")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit  = 100

    q_DA = { t: model.addVar(lb=0.0, ub=P_max, name=f"q_DA_{t}") for t in hours}
    delta_p_plus = {
        (w, p, i, t): model.addVar(lb=0.0, name=f"delta_p_plus_{w}_{p}_{i}_{t}")
        for (w, p, i) in scenarios for t in hours
    }
    delta_p_minus = {
        (w, p, i, t): model.addVar(lb=0.0, name=f"delta_p_minus_{w}_{p}_{i}_{t}")
        for (w, p, i) in scenarios for t in hours
    }

    # Objective: maximize expected profit over all scenarios and hours
    obj = gp.quicksum(
        prob[(w, p, i)] * ( lda[(w, p, i, t)] * q_DA[t]
            + lambda_up[(w, p, i, t)] * delta_p_plus[(w, p, i, t)]
            - lambda_down[(w, p, i, t)] * delta_p_minus[(w, p, i, t)]
        )
        for (w, p, i) in scenarios for t in hours
    )
    model.setObjective(obj, GRB.MAXIMIZE)

    # Imbalance decomposition:
    for (w, p, i) in scenarios:
        for t in hours:
            model.addConstr(
                delta_p_plus[(w, p, i, t)] - delta_p_minus[(w, p, i, t)]
                == p_wind[(w, p, i, t)] - q_DA[t],
                name=f"imbalance_{w}_{p}_{i}_{t}"
            )

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Task 1.1: solver status {model.Status}")

    # Extract results
    q_DA_opt   = {t: q_DA[t].X for t in hours}
    delta_p_plus_opt  = {(w, p, i, t): delta_p_plus[(w, p, i, t)].X
                         for (w, p, i) in scenarios for t in hours}
    delta_p_minus_opt = {(w, p, i, t): delta_p_minus[(w, p, i, t)].X
                         for (w, p, i) in scenarios for t in hours}

    # Profit per combined scenario (sum over all 24 hours)
    profit_scenario = {}
    for (w, p, i) in scenarios:
        total = sum( lda[(w, p, i, t)] * q_DA_opt[t]
            + lambda_up[(w, p, i, t)] * delta_p_plus_opt[(w, p, i, t)]
            - lambda_down[(w, p, i, t)] * delta_p_minus_opt[(w, p, i, t)]
            for t in hours
        )
        profit_scenario[(w, p, i)] = total

    expected_profit = sum(
        prob[sc_key] * profit_scenario[sc_key] for sc_key in scenarios
    )

    return {
        "q_DA":            q_DA_opt,
        "expected_profit": expected_profit,
        "profit_scenario": profit_scenario,
        "delta_p_plus":    delta_p_plus_opt,
        "delta_p_minus":   delta_p_minus_opt,
        "obj_val":         model.ObjVal,
        "scheme":          scheme or sc.get("scheme", "one_price"),
    }

def Ex_post_analysis(data, sc_one, sc_two, n_folds=8, seed=42):
    """
    Task 1.3 — Ex-post analysis via k-fold cross-validation.

    Splits all scenarios into n_folds equal groups. For each fold, optimizes
    using in-sample scenarios, then evaluates the fixed q_DA on out-of-sample
    scenarios. Returns a DataFrame with per-fold IS and OOS expected profits
    for both one-price and two-price schemes.
    """
    hours         = data["hours"]
    all_scenarios = sc_one["scenarios"]
    fold_size     = len(all_scenarios) // n_folds

    shuffled = all_scenarios.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    folds = [shuffled[k * fold_size:(k + 1) * fold_size] for k in range(n_folds)]

    records = []
    for k, in_sc in enumerate(folds):
        oos_sc = [s for j, fold in enumerate(folds) if j != k for s in fold]

        for sc_full, scheme_label in [(sc_one, "one_price"), (sc_two, "two_price")]:
            sc_in  = filter_scenarios(sc_full, in_sc)
            sc_oos = filter_scenarios(sc_full, oos_sc)

            res        = Stochastic_offering_strategy(data, sc_in)
            profit_oos = evaluate_profit(res["q_DA"], sc_oos, hours)

            records.append({
                "fold":                 k + 1,
                "scheme":               scheme_label,
                "in_sample_profit":     res["expected_profit"],
                "out_of_sample_profit": profit_oos,
            })

    return pd.DataFrame(records)


def Risk_averse_offering_strategy(data, sc, alpha=0.90, beta=0.5):
    """
    Task 1.4 — Risk-averse stochastic offering strategy using CVaR.

    Maximizes: (1 - beta) * E[pi] + beta * CVaR_alpha(pi)

    CVaR_alpha is the expected profit in the worst (1-alpha) fraction of
    scenarios. LP formulation via auxiliary variables zeta (VaR) and eta_s.
    """
    hours       = data["hours"]
    P_max       = data["P_max"]
    scenarios   = sc["scenarios"]
    prob        = sc["prob"]
    p_wind      = sc["p_wind"]
    lda         = sc["lda"]
    lambda_up   = sc["lambda_up"]
    lambda_down = sc["lambda_down"]

    model = gp.Model("Risk_averse_offering_strategy")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit  = 300

    # First-stage: DA offer per hour
    q_DA = {t: model.addVar(lb=0.0, ub=P_max, name=f"q_DA_{t}") for t in hours}

    # Second-stage: imbalance variables
    delta_p_plus = {
        (w, p, i, t): model.addVar(lb=0.0, name=f"dp+_{w}_{p}_{i}_{t}")
        for (w, p, i) in scenarios for t in hours
    }
    delta_p_minus = {
        (w, p, i, t): model.addVar(lb=0.0, name=f"dp-_{w}_{p}_{i}_{t}")
        for (w, p, i) in scenarios for t in hours
    }

    # CVaR auxiliary variables
    zeta = model.addVar(lb=-GRB.INFINITY, name="zeta")          # VaR (can be negative)
    eta  = {s: model.addVar(lb=0.0, name=f"eta_{s[0]}_{s[1]}_{s[2]}") for s in scenarios}

    # Expected profit linear expression
    expected_profit_expr = gp.quicksum(
        prob[(w, p, i)] * ( lda[(w, p, i, t)] * q_DA[t]
            + lambda_up[(w, p, i, t)] * delta_p_plus[(w, p, i, t)]
            - lambda_down[(w, p, i, t)] * delta_p_minus[(w, p, i, t)]
        )
        for (w, p, i) in scenarios for t in hours
    )

    # CVaR expression: zeta - 1/(1-alpha) * E[eta_s]
    cvar_expr = zeta - (1.0 / (1.0 - alpha)) * gp.quicksum(
        prob[(w, p, i)] * eta[(w, p, i)] for (w, p, i) in scenarios
    )

    model.setObjective(
        (1.0 - beta) * expected_profit_expr + beta * cvar_expr,
        GRB.MAXIMIZE
    )

    # Imbalance balance constraints
    for (w, p, i) in scenarios:
        for t in hours:
            model.addConstr(
                delta_p_plus[(w, p, i, t)] - delta_p_minus[(w, p, i, t)]
                == p_wind[(w, p, i, t)] - q_DA[t]
            )

    # CVaR constraints: eta_s >= zeta - pi_s  for all scenarios s
    for (w, p, i) in scenarios:
        pi_s = gp.quicksum(
            lda[(w, p, i, t)]           * q_DA[t]
            + lambda_up[(w, p, i, t)]   * delta_p_plus[(w, p, i, t)]
            - lambda_down[(w, p, i, t)] * delta_p_minus[(w, p, i, t)]
            for t in hours
        )
        model.addConstr(eta[(w, p, i)] >= zeta - pi_s)

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Risk_averse_offering_strategy: solver status {model.Status}")

    # Extract optimal q_DA and compute profits analytically
    q_DA_opt = {t: q_DA[t].X for t in hours}

    profit_scenario = {}
    for (w, p, i) in scenarios:
        profit_scenario[(w, p, i)] = sum(
            lda[(w, p, i, t)]           * q_DA_opt[t]
            + lambda_up[(w, p, i, t)]   * max(p_wind[(w, p, i, t)] - q_DA_opt[t], 0.0)
            - lambda_down[(w, p, i, t)] * max(q_DA_opt[t] - p_wind[(w, p, i, t)], 0.0)
            for t in hours
        )

    exp_profit = sum(prob[s] * profit_scenario[s] for s in scenarios)

    # Compute CVaR from optimal zeta*
    zeta_opt = zeta.X
    cvar = zeta_opt - (1.0 / (1.0 - alpha)) * sum(
        prob[s] * max(zeta_opt - profit_scenario[s], 0.0) for s in scenarios
    )

    return {
        "q_DA":            q_DA_opt,
        "expected_profit": exp_profit,
        "profit_scenario": profit_scenario,
        "cvar":            cvar,
        "zeta":            zeta_opt,
        "obj_val":         model.ObjVal,
        "scheme":          sc.get("scheme", ""),
        "beta":            beta,
        "alpha":           alpha,
    }
