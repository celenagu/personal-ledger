# main.py

from pathlib import Path
import os
import pandas as pd
from functools import cmp_to_key
from formatter import format_csv, format_to_df
# from sheets_uploader import SheetsUploader

def main():
    out_file = "files/formatted.csv"

    # Merge chequing/savings/credit if present
    candidates = [
        "files/chequing.csv",
        "files/savings.csv",
        "files/credit.csv",
    ]
    existing = [p for p in candidates if Path(p).exists()]

    if existing:
        frames = []
        for p in existing:
            df = format_to_df(p).copy()
            df["Source"] = Path(p).name
            df["OrigOrder"] = range(len(df))
            frames.append(df)
        new_df = pd.concat(frames, ignore_index=True)
    else:
        in_file = "files/input.csv"
        new_df = format_to_df(in_file)
        new_df["Source"] = Path(in_file).name
        new_df["OrigOrder"] = range(len(new_df))

    # Append-only master with duplicate handling
    # Keep existing rows; from the new batch, add only the surplus copies beyond what is already in master
    if Path(out_file).exists():
        master_df = pd.read_csv(out_file)
        master_df["Source"] = "master"
        master_df["OrigOrder"] = range(len(master_df))
        # Identity keys: Date, Name, Type, SignedAmount
        def with_signed_amount(df):
            d = df.copy()
            d["Out"] = pd.to_numeric(d["Out"], errors="coerce")
            d["In"] = pd.to_numeric(d["In"], errors="coerce")
            d["SignedAmount"] = (d["In"].fillna(0) - d["Out"].fillna(0)).round(2)
            return d

        master_k = with_signed_amount(master_df)
        new_k = with_signed_amount(new_df)

        keys = ["Date", "Name", "Type", "SignedAmount"]

        # Count occurrences in master for each key
        master_counts = (
            master_k.groupby(keys).size().rename("count_master").reset_index()
        )
        # For each new row, compute its position within identical-key rows 
        tmp = new_k.copy()
        tmp["_dup_index"] = tmp.groupby(keys).cumcount()
        tmp = tmp.merge(master_counts, on=keys, how="left")
        tmp["count_master"] = tmp["count_master"].fillna(0).astype(int)
        # Keep only the rows whose index exceeds existing count in master
        new_only_k = tmp[tmp["_dup_index"] >= tmp["count_master"]].drop(columns=["_dup_index", "count_master"]) 
        # Remove helper column
        new_only = new_only_k.drop(columns=["SignedAmount"], errors="ignore")
        if os.getenv("TXN_DEBUG"):
            skipped = len(new_df) - len(new_only)
            print(f"DEBUG: new={len(new_df)} added={len(new_only)} skipped_as_existing={skipped}")
        combined = pd.concat([master_df, new_only], ignore_index=True)
    else:
        combined = new_df


    # - Primary: Date ascending
    # - Tie: if same Source, keep original row order (OrigOrder) otherwise order by Name ascending
    def _cmp(a, b):
        if a["Date"] != b["Date"]:
            return -1 if a["Date"] < b["Date"] else 1
        if a["Source"] == b["Source"]:
            if a["OrigOrder"] == b["OrigOrder"]:
                return 0
            return -1 if a["OrigOrder"] < b["OrigOrder"] else 1
        # Different sources on same date: order by Name
        an, bn = str(a["Name"]), str(b["Name"]) 
        if an == bn:
            # Fallback 
            return -1 if a["Source"] < b["Source"] else 1
        return -1 if an < bn else 1

    records = combined.to_dict("records")
    records_sorted = sorted(records, key=cmp_to_key(_cmp))
    combined_sorted = pd.DataFrame.from_records(records_sorted, columns=combined.columns)
    combined_sorted = combined_sorted.drop(columns=["Source", "OrigOrder"], errors="ignore")
    combined_sorted.to_csv(out_file, index=False)
    print("Merged and saved to:", out_file)
    # upload_csv_to_sheets(out_file, "Jan 2025 Report")

if __name__ == "__main__":                                                
    import traceback, sys                                                 
    try:                                                                  
        main()                                                            
    except Exception:                                                     
        traceback.print_exc()                                             
        input("Error occurred. Press Enter to exit...")                   
        sys.exit(1)      
