from itertools import product


def filter_scenarios(sc, scenario_subset):
    """Return a new scenario dict restricted to the given (w, p, i) tuples."""
    subset = list(scenario_subset)
    n = len(subset)
    hours = sorted({key[3] for key in sc["p_wind"]})
    return {
        "scenarios":   subset,
        "prob":        {s: 1.0 / n for s in subset},
        "p_wind":      {(w,p,i,t): sc["p_wind"][(w,p,i,t)]      for (w,p,i) in subset for t in hours},
        "lda":         {(w,p,i,t): sc["lda"][(w,p,i,t)]         for (w,p,i) in subset for t in hours},
        "lb":          {(w,p,i,t): sc["lb"][(w,p,i,t)]          for (w,p,i) in subset for t in hours},
        "SI":          {(w,p,i,t): sc["SI"][(w,p,i,t)]          for (w,p,i) in subset for t in hours},
        "lambda_up":   {(w,p,i,t): sc["lambda_up"][(w,p,i,t)]   for (w,p,i) in subset for t in hours},
        "lambda_down": {(w,p,i,t): sc["lambda_down"][(w,p,i,t)] for (w,p,i) in subset for t in hours},
        "scheme":      sc["scheme"],
        "n_scenarios": n,
    }


def build_scenarios(data, scheme):
    """
    Build the full combined scenario dictionary for the LP model.

    Parameters
    ----------
    data   : dict returned by input_data()
    scheme : str, either 'one_price' or 'two_price'

    Returns
    -------
    scenarios_data : dict
        scenarios    : list[tuple]  all (w, p, i) combinations — 1600 total
        prob         : dict  (w, p, i)      -> float  equal weight = 1/1600
        p_wind       : dict  (w, p, i, t)   -> float  MW
        lda          : dict  (w, p, i, t)   -> float  DA price EUR/MWh
        lb           : dict  (w, p, i, t)   -> float  balancing price EUR/MWh
        SI           : dict  (w, p, i, t)   -> int    {0, 1}
        alpha_plus   : dict  (w, p, i, t)   -> float  settlement price for surplus
        alpha_minus  : dict  (w, p, i, t)   -> float  settlement price for deficit
        scheme       : str   scheme used
        n_scenarios  : int   1600
    """
    if scheme not in ("one_price", "two_price"):
        raise ValueError(f"scheme must be 'one_price' or 'two_price', got '{scheme}'")

    wind_df  = data["wind_scenarios"]
    price_df = data["price_scenarios"]
    imb_df   = data["imbalance_scenarios"]
    hours    = data["hours"]

    DEFICIT_FACTOR = 1.25
    SURPLUS_FACTOR = 0.85

    scenarios    = list(product(data["wind_scen"], data["price_scen"], data["imb_scen"]))
    n_scenarios  = len(scenarios)
    prob         = {sc: 1 / n_scenarios for sc in scenarios}

    p_wind_dict      = {}
    lda_dict         = {}
    lb_dict          = {}
    SI_dict          = {}
    lambda_up_dict  = {}
    lambda_down_dict = {}

    for (w, p, i) in scenarios:
        for t in hours:
            key = (w, p, i, t)

            pw  = float(wind_df.loc[t, w])
            lda = float(price_df.loc[t, p])
            si  = int(imb_df.loc[t, i])
            lb  = DEFICIT_FACTOR * lda if si == 1 else SURPLUS_FACTOR * lda

            # Settlement prices depend on scheme
            if scheme == "one_price":
                l_up = lb
                l_down = lb

            else:  # two_price
                if si == 1:    # system short
                    l_up = lda
                    l_down = lb
                else:          # system long
                    l_up = lb
                    l_down = lda

            p_wind_dict[key]      = pw
            lda_dict[key]         = lda
            lb_dict[key]          = lb
            SI_dict[key]          = si
            lambda_up_dict[key]  = l_up
            lambda_down_dict[key] = l_down

    return {
        "scenarios":   scenarios,
        "prob":        prob,
        "p_wind":      p_wind_dict,
        "lda":         lda_dict,
        "lb":          lb_dict,
        "SI":          SI_dict,
        "lambda_up":  lambda_up_dict,
        "lambda_down": lambda_down_dict,
        "scheme":      scheme,
        "n_scenarios": n_scenarios,
    }
