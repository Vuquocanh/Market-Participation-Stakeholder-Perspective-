import pandas as pd


def print_comparison(res_one, res_two):
    print(f"\n{'='*60}")
    print("  Comparison")
    print(f"{'='*60}")
    diff = res_two["expected_profit"] - res_one["expected_profit"]
    print(f"  One-price expected profit : {res_one['expected_profit']:>12,.2f} EUR")
    print(f"  Two-price expected profit : {res_two['expected_profit']:>12,.2f} EUR")
    print(f"  Difference (two - one)    : {diff:>+12,.2f} EUR")

    hours = sorted(res_one["q_DA"].keys())
    print(f"\n  DA offer comparison (MW):")
    rows = []
    for t in hours:
        rows.append({
            "Hour": t,
            "q_DA one-price": round(res_one["q_DA"][t], 3),
            "q_DA two-price": round(res_two["q_DA"][t], 3),
            "Diff": round(res_two["q_DA"][t] - res_one["q_DA"][t], 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def print_results(res, label):
    scheme = res["scheme"]
    print(f"\n{'='*60}")
    print(f"  {label}  ({scheme})")
    print(f"{'='*60}")
    print(f"  Expected profit : {res['expected_profit']:>12,.2f} EUR")
    print(f"  LP obj value    : {res['obj_val']:>12,.2f} EUR")

    print(f"\n  Optimal DA offers (MW):")
    q = res["q_DA"]
    hours = sorted(q.keys())
    rows = [{"Hour": t, "q_DA (MW)": round(q[t], 3)} for t in hours]
    print(pd.DataFrame(rows).to_string(index=False))

    profits = res["profit_scenario"]
    p_vals = list(profits.values())
    print(f"\n  Scenario profit stats (EUR):")
    print(f"    min  : {min(p_vals):>12,.2f}")
    print(f"    mean : {sum(p_vals)/len(p_vals):>12,.2f}")
    print(f"    max  : {max(p_vals):>12,.2f}")
