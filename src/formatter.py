# formatter.py

import pandas as pd
from pathlib import Path

def infer_type_from_name(in_file):
    name = Path(in_file).name.lower()
    if "chequing" in name:
        return "Chequing"
    elif "savings" in name:
        return "Savings"
    elif "credit" in name:
        return "Credit"
    else:
        return "Uncategorized"

def format_to_df(in_file):
    df = pd.read_csv(in_file, header=None)
    account_type = infer_type_from_name(in_file)

    df = df.dropna(how='all')   # Remove empty rows

    df.columns = ["Date", "Name", "Out", "In", "Total"]

    df["Date"] = pd.to_datetime(df["Date"], errors='coerce').dt.strftime('%Y-%m-%d')
    df["Name"] = (
        df["Name"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    df.insert(2, "Type", account_type)
    df.insert(0, "Category", "Uncategorized")
    return df

def format_csv(in_file, out_file):
    df = format_to_df(in_file)
    df.to_csv(out_file, index=False)
    print("Formatted and saved to:", out_file)
