# formatter.py

from pathlib import Path
import pandas as pd
from functools import cmp_to_key
from errors import LedgerError

# Infer account type from filename
def infer_type_from_name(in_file):
    name = Path(in_file).name.lower()
    if "chequing" in name:
        return "Chequing"
    if "savings" in name:
        return "Savings"
    if "credit" in name:
        return "Credit"
    return "Uncategorized"

# Format CSV to standardized DataFrame
def format_to_df(in_file):
    # Read CSV without header
    try:
        df = pd.read_csv(in_file, header=None)
    except pd.errors.EmptyDataError:
        raise LedgerError(f"{in_file} is empty. Re-export it from the bank or delete it.")
    except pd.errors.ParserError as e:
        raise LedgerError(f"Could not parse {in_file} as CSV: {e}")
    account_type = infer_type_from_name(in_file)

    # Drop completely empty rows
    df = df.dropna(how="all")

    # TD exports have exactly 5 columns with no header row
    if df.shape[1] != 5:
        raise LedgerError(
            f"{in_file} has {df.shape[1]} columns, expected 5 (Date, Name, Out, In, Total). "
            "Is it a TD CSV export?"
        )

    # Standardize to 5 columns: Date, Name, Out, In, Total
    df.columns = ["Date", "Name", "Out", "In", "Total"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Name"] = (
        df["Name"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    df.insert(2, "Type", account_type)
    df.insert(0, "Category", "Uncategorized")
    return df

# Merge multiple input files into a single DataFrame
def merge_input_files(
    candidates=None,
    default_file="files/input/input.csv",
):
    if candidates is None:
        candidates = [
            "files/input/chequing.csv",
            "files/input/savings.csv",
            "files/input/credit.csv",
        ]
    existing = [p for p in candidates if Path(p).exists()]

    # If multiple files exist, merge them
    if existing:
        frames = []
        for p in existing:
            df = format_to_df(p).copy()
            df["Source"] = Path(p).name
            df["OrigOrder"] = range(len(df))
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    if not Path(default_file).exists():
        raise LedgerError(
            "No input files found. Put at least one of these in place: "
            + ", ".join(candidates + [default_file])
        )
    df = format_to_df(default_file)
    df["Source"] = Path(default_file).name
    df["OrigOrder"] = range(len(df))
    return df

# Convert series to numeric, handling common formatting issues
def to_num(series):
    if series is None:
        return pd.Series(dtype=float)
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")

# Normalize date series, handling various formats including Excel serial dates
def normalize_date_series(series):
    raw = series.astype(str).str.strip()
    parsed = pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(raw, errors="coerce")
    excel_mask = numeric.notna() & raw.str.fullmatch(r"\d+(\.\d+)?", na=False)
    if excel_mask.any():
        excel_dates = pd.to_datetime(
            numeric,
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
        parsed = parsed.where(~excel_mask, excel_dates)
    formatted = parsed.dt.strftime("%Y-%m-%d")
    formatted = formatted.where(parsed.notna(), raw)
    return formatted

# Map a sheet header name to the standard column name (unknown names are kept as-is)
def canonical_column(col):
    key = str(col).strip().lower()
    if key in {"date", "transaction date"}:
        return "Date"
    if key in {"name", "description", "merchant"}:
        return "Name"
    if key in {"type", "account", "account type"}:
        return "Type"
    if key in {"out", "debit", "withdrawal"}:
        return "Out"
    if key in {"in", "credit", "deposit"}:
        return "In"
    if key in {"amount", "signedamount", "signed amount"}:
        return "SignedAmount"
    if key in {"category"}:
        return "Category"
    if key in {"total", "balance"}:
        return "Total"
    return col

# Normalize history DataFrame columns
def normalize_history_df(df):
    if df is None or df.empty:
        return df
    df = df.rename(columns={col: canonical_column(col) for col in df.columns})

    if "Name" in df.columns:
        df["Name"] = (
            df["Name"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
    if "Date" in df.columns:
        df["Date"] = normalize_date_series(df["Date"])
    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str).str.strip()
    return df

# Add OverallTotal column based on account types
def add_overall_total(df):
    if df is None or df.empty or "Type" not in df.columns:
        return df
    totals = df["Total"] if "Total" in df.columns else pd.Series([None] * len(df))
    totals = to_num(totals)
    balances = {"Chequing": None, "Savings": None}
    overall_totals = []
    for idx, row in df.iterrows():
        account_type = row.get("Type")
        if account_type in balances:
            current = totals.iloc[idx]
            if pd.notna(current):
                balances[account_type] = current
        if account_type in balances:
            known = [v for v in balances.values() if v is not None]
            overall_total = sum(known) if known else None
        else:
            overall_total = None
        overall_totals.append(overall_total)
    df = df.copy()
    df["OverallTotal"] = overall_totals
    return df

# Attach OverallTotal to new_only_df from combined_sorted
def attach_overall_total(new_only_df, combined_sorted):
    if (
        new_only_df is None
        or new_only_df.empty
        or "Source" not in new_only_df.columns
        or "OrigOrder" not in new_only_df.columns
    ):
        return new_only_df
    totals_cols = ["Source", "OrigOrder", "OverallTotal"]
    return new_only_df.merge(
        combined_sorted[totals_cols],
        on=["Source", "OrigOrder"],
        how="left",
    )

# Format input CSV and save to output CSV
def format_csv(in_file, out_file):
    df = format_to_df(in_file)
    df.to_csv(out_file, index=False)
    print("Formatted and saved to:", out_file)

# Sort transactions by Date, then by Source and Name
def sort_transactions(df):
    # - Primary: Date ascending
    # - Tie: if same source, keep original row order (OrigOrder) otherwise order by Name ascending
    def _cmp(a, b):
        if a["Date"] != b["Date"]:
            return -1 if a["Date"] < b["Date"] else 1
        if a.get("Source") == b.get("Source"):
            if a.get("OrigOrder") == b.get("OrigOrder"):
                return 0
            return -1 if a.get("OrigOrder") < b.get("OrigOrder") else 1
        an, bn = str(a.get("Name")), str(b.get("Name"))
        if an == bn:
            return -1 if str(a.get("Source")) < str(b.get("Source")) else 1
        return -1 if an < bn else 1

    records = df.to_dict("records")
    records_sorted = sorted(records, key=cmp_to_key(_cmp))
    return pd.DataFrame.from_records(records_sorted, columns=df.columns)

# Deduplicate new rows against master DataFrame
def dedupe_new_rows(new_df, master_df):
    new_only = new_df
    if master_df is None:
        return new_df, new_df

    master_df = master_df.copy()
    master_df["Source"] = "master"
    master_df["OrigOrder"] = range(len(master_df))

    def _normalize_amount_cols(df, invert=False):
        d = df.copy()
        if "Out" not in d.columns:
            d["Out"] = 0
        if "In" not in d.columns:
            d["In"] = 0
        d["Out"] = to_num(d["Out"]).fillna(0)
        d["In"] = to_num(d["In"]).fillna(0)
        if "Total" in d.columns:
            d["Total"] = to_num(d["Total"]).fillna(0)
        if invert:
            d["In"], d["Out"] = d["Out"], d["In"]
        return d

    new_k = _normalize_amount_cols(new_df)
    master_k = _normalize_amount_cols(master_df)
    master_k_alt = _normalize_amount_cols(master_df, invert=True)

    keys = ["Date", "Name", "Type", "In", "Out", "Total"]
    keys = [k for k in keys if k in master_k.columns and k in new_k.columns]
    if not keys:
        combined = pd.concat([master_df, new_df], ignore_index=True)
        return combined, new_df

    master_keys = master_k[keys].drop_duplicates()
    master_keys_alt = master_k_alt[keys].drop_duplicates()
    matched = new_k.merge(master_keys, on=keys, how="inner")
    matched_alt = new_k.merge(master_keys_alt, on=keys, how="inner")
    if len(matched_alt) > len(matched):
        master_k = master_k_alt

    master_counts = (
        master_k.groupby(keys).size().rename("count_master").reset_index()
    )
    tmp = new_k.copy()
    tmp["_dup_index"] = tmp.groupby(keys).cumcount()
    tmp = tmp.merge(master_counts, on=keys, how="left")
    tmp["count_master"] = tmp["count_master"].fillna(0).astype(int)
    new_only_k = tmp[tmp["_dup_index"] >= tmp["count_master"]].drop(
        columns=["_dup_index", "count_master"]
    )
    new_only = new_only_k.drop(columns=["SignedAmount"], errors="ignore")

    if new_only.empty or new_only.isna().all().all():
        combined = master_df
    elif master_df.empty or master_df.isna().all().all():
        combined = new_only
    else:
        combined = pd.concat([master_df, new_only], ignore_index=True)

    return combined, new_only


