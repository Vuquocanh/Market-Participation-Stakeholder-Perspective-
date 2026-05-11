from pathlib import Path

from data_loader import input_data
from scenario_builder import build_scenarios
from model.Step_1 import Stochastic_offering_strategy, Ex_post_analysis, Risk_averse_offering_strategy
from model.Step_2 import ALSO_X, CVaR_approach, analyze_p90_tradeoff, verify_p90_requirement
from load_profile_generator import generate_load_profiles, split_profiles
from display import print_results, print_comparison
from export_results import save_results, save_task2_1_results, save_task2_2_results, save_task2_3_results
from results_config import structure_task1, structure_task2_1, structure_task2_2, structure_task2_3

RESULTS_DIR = Path(__file__).parent / "results"

# =============================================================================
# TASK 1.1 — Stochastic offering strategy, one-price scheme
# =============================================================================
def run_task1_1(data):
    print("\n======= Task 1.1 — One-Price =======")
    sc = build_scenarios(data, scheme="one_price")
    print(f"  Total scenarios: {sc['n_scenarios']}")
    res = Stochastic_offering_strategy(data, sc)
    print_results(res, "Task 1.1 — One-Price Scheme")
    save_results(res, structure_task1, RESULTS_DIR / "Task_1_1")
    return res

# =============================================================================
# TASK 1.2 — Stochastic offering strategy, two-price scheme
# =============================================================================
def run_task1_2(data):
    print("\n======= Task 1.2 — Two-Price =======")
    sc = build_scenarios(data, scheme="two_price")
    print(f"  Total scenarios: {sc['n_scenarios']}")
    res = Stochastic_offering_strategy(data, sc)
    print_results(res, "Task 1.2 — Two-Price Scheme")
    save_results(res, structure_task1, RESULTS_DIR / "Task_1_2")
    return res

# =============================================================================
# TASK 1.3 — Ex-post analysis: 8-fold cross-validation
# =============================================================================
def run_task1_3(data):
    print("\n======= Task 1.3 — Ex-post Analysis (8-fold CV) =======")
    sc_one = build_scenarios(data, scheme="one_price")
    sc_two = build_scenarios(data, scheme="two_price")

    df = Ex_post_analysis(data, sc_one, sc_two)

    out_dir = RESULTS_DIR / "Task_1_3"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "cv_results.csv", index=False)

    import pandas as pd
    avg_rows = []
    print(f"\n  {'Scheme':<12}  {'Avg IS profit':>15}  {'Avg OOS profit':>15}")
    print(f"  {'-'*45}")
    for scheme in ("one_price", "two_price"):
        sub = df[df["scheme"] == scheme]
        avg_is  = sub["in_sample_profit"].mean()
        avg_oos = sub["out_of_sample_profit"].mean()
        print(f"  {scheme:<12}  {avg_is:>15,.2f}  {avg_oos:>15,.2f}")
        avg_rows.append({"scheme": scheme, "avg_in_sample_profit": avg_is,
                         "avg_out_of_sample_profit": avg_oos})

    pd.DataFrame(avg_rows).to_csv(out_dir / "cv_averages.csv", index=False)
    
    return df

# =============================================================================
# TASK 1.4 — Risk-averse offering strategy (CVaR efficient frontier)
# =============================================================================
def run_task1_4(data, alpha=0.90,
                betas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    print("\n======= Task 1.4 — Risk-Averse Offering Strategy =======")
    sc_one = build_scenarios(data, scheme="one_price")
    sc_two = build_scenarios(data, scheme="two_price")

    records = []
    for scheme_label, sc in [("one_price", sc_one), ("two_price", sc_two)]:
        for beta in betas:
            print(f"  {scheme_label}, beta={beta:.1f} ...", end=" ", flush=True)
            res = Risk_averse_offering_strategy(data, sc, alpha=alpha, beta=beta)
            records.append({
                "scheme":           scheme_label,
                "beta":             beta,
                "expected_profit":  res["expected_profit"],
                "cvar":             res["cvar"],
            })
            print(f"E[pi]={res['expected_profit']:,.0f}  CVaR={res['cvar']:,.0f}")

    out_dir = RESULTS_DIR / "Task_1_4"
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "efficient_frontier.csv", index=False)

    return df

# =============================================================================
# TASK 2.1 — FCR-D Up reserve bid: ALSO-X vs CVaR
# =============================================================================
def run_task2_1(n_profiles=300, n_in_sample=100):
    print("\n======= Task 2.1 — FCR-D Up Reserve Bid =======")
    profiles = generate_load_profiles(n_profiles=n_profiles)
    in_sample, out_of_sample = split_profiles(profiles, n_in_sample=n_in_sample)
    print(f"  In-sample: {in_sample.shape[0]} profiles, "
          f"Out-of-sample: {out_of_sample.shape[0]} profiles")

    print("  Running ALSO-X ...")
    res_alsox = ALSO_X(in_sample)
    print(f"    c_up = {res_alsox['c_up']:.2f} kW  "
          f"(converged in {res_alsox['iterations']} iter, "
          f"{res_alsox['n_fixed']} pairs fixed)")

    print("  Running CVaR approach ...")
    res_cvar = CVaR_approach(in_sample)
    print(f"    c_up = {res_cvar['c_up']:.2f} kW  "
          f"(beta = {res_cvar['beta']:.4f})")

    save_task2_1_results(
        res_alsox,
        res_cvar,
        profiles,
        n_in_sample,
        structure_task2_1,
        RESULTS_DIR / "Task_2_1",
    )

    return {
        "profiles": profiles,
        "in_sample": in_sample,
        "out_of_sample": out_of_sample,
        "ALSO-X": res_alsox,
        "CVaR": res_cvar,
    }

# =============================================================================
# TASK 2.2 - Out-of-sample verification of P90 requirement
# =============================================================================
def run_task2_2(task2_1_results):
    print("\n======= Task 2.2 - Out-of-Sample P90 Verification =======")
    reserve_bids = {
        "ALSO-X": task2_1_results["ALSO-X"]["c_up"],
        "CVaR": task2_1_results["CVaR"]["c_up"],
    }
    verification = verify_p90_requirement(
        reserve_bids,
        task2_1_results["out_of_sample"],
    )

    print(f"  {'Method':<8}  {'c_up (kW)':>9}  {'η̂^m':>8}  {'η̂^j':>8}  {'ESF (kW)':>9}  {'P90(η̂^m)':>9}  {'P90(η̂^j)':>9}")
    print(f"  {'-'*72}")
    for method, metrics in verification.items():
        print(
            f"  {method:<8}  {metrics['c_up']:>9.2f}"
            f"  {100*metrics['eta_m']:>7.2f}%"
            f"  {100*metrics['eta_j']:>7.2f}%"
            f"  {metrics['esf_kW']:>9.4f}"
            f"  {'met' if metrics['p90_met_eta_m'] else 'FAIL':>9}"
            f"  {'met' if metrics['p90_met_eta_j'] else 'FAIL':>9}"
        )

    save_task2_2_results(
        verification,
        structure_task2_2,
        RESULTS_DIR / "Task_2_2",
    )

    return verification

# =============================================================================
# TASK 2.3 - Energinet perspective: P-requirement sensitivity
# =============================================================================
def run_task2_3(task2_1_results, reliability_levels=None):
    print("\n======= Task 2.3 - P-Requirement Sensitivity =======")
    sweep = analyze_p90_tradeoff(
        task2_1_results["in_sample"],
        task2_1_results["out_of_sample"],
        reliability_levels=reliability_levels,
    )

    print(f"  {'P req.':>8}  {'c_up (kW)':>10}  {'η̂^m':>8}  {'η̂^j':>8}  {'ESF (kW)':>10}")
    print(f"  {'-'*55}")
    for row in sweep:
        print(
            f"  {100 * row['required_reliability']:>7.1f}%  "
            f"{row['c_up_kW']:>10.2f}  "
            f"{100 * row['eta_m']:>7.2f}%  "
            f"{100 * row['eta_j']:>7.2f}%  "
            f"{row['expected_shortfall_kW']:>10.3f}"
        )

    save_task2_3_results(
        sweep,
        structure_task2_3,
        RESULTS_DIR / "Task_2_3",
    )

    return sweep

# =============================================================================
# Entry point — toggle which tasks to run
# =============================================================================
if __name__ == "__main__":

# Step 1

    data = input_data(data_dir="data")
    print(f"  Hours: {len(data['hours'])}, "
          f"Wind scenarios: {len(data['wind_scen'])}, "
          f"Price scenarios: {len(data['price_scen'])}, "
          f"Imbalance scenarios: {len(data['imb_scen'])}")

    res_one = run_task1_1(data)
    res_two = run_task1_2(data)
    print_comparison(res_one, res_two)
    run_task1_3(data)
    run_task1_4(data, alpha=0.90, betas=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Step 2   
 
    task2_1_results = run_task2_1()
    run_task2_2(task2_1_results)
    run_task2_3(task2_1_results)
