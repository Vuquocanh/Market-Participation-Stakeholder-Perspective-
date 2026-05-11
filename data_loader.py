import pandas as pd
from pathlib import Path


def input_data(data_dir="data"):
    """
    Load raw input data from CSV files.

    Returns
    -------
    data : dict
        wind_scenarios      : DataFrame (24 x 20)  MW,       index=time_id
        price_scenarios     : DataFrame (24 x 20)  EUR/MWh,  index=time_id
        imbalance_scenarios : DataFrame (24 x  4)  {0,1},    index=time_id
        hours               : list[int]  1..24
        wind_scen           : list[str]  ['w1'..'w20']
        price_scen          : list[str]  ['p1'..'p20']
        imb_scen            : list[str]  ['i1'..'i4']
        P_max               : float  installed wind capacity (MW)
    """
    data_dir = Path(data_dir)

    wind_df  = pd.read_csv(data_dir / "wind_scenarios.csv",      index_col="time_id")
    price_df = pd.read_csv(data_dir / "da_price_scenarios.csv",  index_col="time_id")
    imb_df   = pd.read_csv(data_dir / "imbalance_scenarios.csv", index_col="time_id")

    data = {
        "wind_scenarios":      wind_df,
        "price_scenarios":     price_df,
        "imbalance_scenarios": imb_df,
        "hours":               wind_df.index.tolist(),
        "wind_scen":           wind_df.columns.tolist(),
        "price_scen":          price_df.columns.tolist(),
        "imb_scen":            imb_df.columns.tolist(),
        "P_max":               500.0,
    }

    return data
