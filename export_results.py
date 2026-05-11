import pandas as pd
from pathlib import Path


def build_task2_1_results(res_alsox, res_cvar, profiles, n_in_sample):
    """Build CSV-ready task 2.1 reserve bid and load profile records."""
    profile_rows = []
    for profile_idx, profile in enumerate(profiles, start=1):
        sample = "in_sample" if profile_idx <= n_in_sample else "out_of_sample"
        for minute, load_kw in enumerate(profile, start=1):
            profile_rows.append({
                "profile_id": profile_idx,
                "sample": sample,
                "minute": minute,
                "load_kW": load_kw,
                "available_flex_kW": load_kw - 220.0,
            })

    return {
        "reserve_bids": [{
            "method": "ALSO-X",
            "c_up_kW": res_alsox["c_up"],
            "iterations": res_alsox["iterations"],
            "n_fixed": res_alsox["n_fixed"],
            "beta": "",
        }, {
            "method": "CVaR",
            "c_up_kW": res_cvar["c_up"],
            "iterations": "",
            "n_fixed": "",
            "beta": res_cvar["beta"],
        }],
        "load_profiles": profile_rows,
    }


def save_task2_1_results(res_alsox, res_cvar, profiles, n_in_sample, structure, base_path):
    """Save all task 2.1 CSV outputs using the configured export structure."""
    results = build_task2_1_results(res_alsox, res_cvar, profiles, n_in_sample)
    save_results(results, structure, base_path)
    return results


def build_task2_2_results(verification):
    """Build CSV-ready task 2.2 out-of-sample verification records."""
    rows = []
    for method, metrics in verification.items():
        row = {"method": method}
        row.update(metrics)
        row["p90_met_eta_m"] = bool(row["p90_met_eta_m"])
        row["p90_met_eta_j"] = bool(row["p90_met_eta_j"])
        rows.append(row)

    return {"p90_verification": rows}


def save_task2_2_results(verification, structure, base_path):
    """Save task 2.2 CSV outputs using the configured export structure."""
    results = build_task2_2_results(verification)
    save_results(results, structure, base_path)
    return results


def save_task2_3_results(sweep_records, structure, base_path):
    """Save task 2.3 CSV outputs using the configured export structure."""
    results = {"p_requirement_sweep": sweep_records}
    save_results(results, structure, base_path)
    return results


def save_results(result_dict, structure_dict, base_path):
    """Save all results in result_dict to CSV files under base_path,
    using the layout defined in structure_dict."""

    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    for key, config in structure_dict.items():

        if key not in result_dict:
            continue

        data = result_dict[key]

        try:
            if config["dim"] == "records":
                df = pd.DataFrame(data)

            elif config["dim"] == 0:
                df = pd.DataFrame([data], columns=[config["filename"].replace(".csv", "")])

            elif config["dim"] == 1:
                if isinstance(data, dict):
                    rows = list(data.items())
                elif isinstance(data, (list, tuple)):
                    rows = list(enumerate(data, start=1))
                else:
                    rows = [(1, data)]
                df = pd.DataFrame(rows, columns=config["columns"])

            elif config["dim"] == 2:
                df = pd.DataFrame(
                    [(k1, k2, v) for (k1, k2), v in data.items()],
                    columns=config["columns"]
                )

            elif config["dim"] == 3:
                df = pd.DataFrame(
                    [(k1, k2, k3, v) for (k1, k2, k3), v in data.items()],
                    columns=config["columns"]
                )

            elif config["dim"] == 4:
                df = pd.DataFrame(
                    [(k1, k2, k3, k4, v) for (k1, k2, k3, k4), v in data.items()],
                    columns=config["columns"]
                )

            else:
                raise ValueError(f"Unsupported dimension {config['dim']} for key '{key}'")

            df.to_csv(base_path / config["filename"], index=False)

        except Exception as e:
            print(f"ERROR saving key='{key}': {e}")
            raise
