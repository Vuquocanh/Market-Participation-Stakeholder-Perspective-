import pandas as pd
from pathlib import Path

excel_file = "Input_data.xlsx"
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

xls = pd.ExcelFile(excel_file)

sheet_configs = {
    "Wind_scenarios":    {"index_col": "time_id"},
    "DA_price_scenarios":{"index_col": "time_id"},
    "Imbalance_scenarios":{"index_col": "time_id"},
}

for sheet, cfg in sheet_configs.items():
    df = pd.read_excel(xls, sheet_name=sheet, header=0)

    if cfg["index_col"] and cfg["index_col"] in df.columns:
        df = df.set_index(cfg["index_col"])

    csv_name = sheet.lower() + ".csv"
    csv_path = output_dir / csv_name
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}  {df.shape}")

print("\nDone. Files in ./data/")
