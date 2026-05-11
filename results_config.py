"""
results_config.py
Defines the export structure for each task's results.
    dim=0 — scalar
    dim=1 — dict {id: value}
    dim=3 — dict {(id1, id2, id3): value}
"""

# Task 1.1 / 1.2 — Stochastic offering strategy (one-price or two-price)
structure_task1 = {
    "expected_profit": {"dim": 0, "filename": "expected_profit.csv"},
    "obj_val":         {"dim": 0, "filename": "obj_val.csv"},
    "q_DA":            {"dim": 1, "columns": ["hour", "q_DA_MW"],
                        "filename": "q_DA.csv"},
    "profit_scenario": {"dim": 3, "columns": ["wind_scen", "price_scen", "imb_scen", "profit_EUR"],
                        "filename": "profit_scenario.csv"},
}


# Task 2.1 - FCR-D Up reserve bids from ALSO-X and CVaR
structure_task2_1 = {
    "reserve_bids": {"dim": "records", "filename": "reserve_bids.csv"},
    "load_profiles": {"dim": "records", "filename": "load_profiles.csv"},
}


# Task 2.2 - out-of-sample P90 verification
structure_task2_2 = {
    "p90_verification": {"dim": "records", "filename": "p90_verification.csv"},
}


# Task 2.3 - P-requirement sweep using ALSO-X
structure_task2_3 = {
    "p_requirement_sweep": {"dim": "records", "filename": "p_requirement_sweep.csv"},
}
