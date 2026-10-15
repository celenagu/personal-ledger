# main.py

# Import necessary modules
from pathlib import Path
import os
import json
import pandas as pd
from formatter import (
    add_overall_total,
    attach_overall_total,
    dedupe_new_rows,
    merge_input_files,
    normalize_date_series,
    normalize_history_df,
    sort_transactions,
)
from categorizer import (
    apply_merchant_map,
    apply_tfidf_classifier,
    is_uncategorized,
    record_assignments,
    update_merchant_map_from_history,
)
from llm_categorizer import apply_llm_classifier
from sheets_uploader import SheetsUploader
from errors import LedgerError
from googleapiclient.errors import HttpError

# All paths (files/, secrets/) are relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==============================
# Helper Functions
# ==============================

# Helper function for configuration and environment variable handling
def _read_gsheets_config(path="secrets/gsheets_config.json"):
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}

# Interpret truth values
def _truthy(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False

# check if config key is enabled
def _config_enabled(config, key):
    return _truthy(config.get(key))

# check if environment variable is enabled
def _env_enabled(name):
    return os.getenv(name) == "1"

# ==============================
# Main Functionality
# ==============================

# main function to process and upload data
def main():
    # Run from the project root so relative paths work no matter where the script is launched from
    os.chdir(PROJECT_ROOT)
    out_file = "files/formatted.csv"

    # Merge input files
    new_df = merge_input_files()        # df = pd.DataFrame

    # Note: what is a data frame?
    # 2D, size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns).

    # Read current google sheets config
    config = _read_gsheets_config()
    sync_enabled = _env_enabled("SHEETS_SYNC") or _config_enabled(config, "sync_enabled")
    history_enabled = _env_enabled("SHEETS_HISTORY") or _config_enabled(config, "history_enabled") or sync_enabled
    sheets_history = None

    # Read history from Google Sheets if enabled
    if history_enabled:
        sheets_history = SheetsUploader().read_sheet()
        sheets_history = normalize_history_df(sheets_history)   
        if sheets_history is not None and sheets_history.empty:
            sheets_history = None

    # Update merchant map and apply categorization from sheet history if exists
    if sheets_history is None:
        update_merchant_map_from_history()
    else:
        update_merchant_map_from_history(history_df=sheets_history)

    # apply merchant map, then LLM (if enabled), then TF-IDF for anything still uncategorized
    # Each step's results are logged with their source: TF-IDF guesses are never learned as rules,
    # and any category later changed in the sheet is recognized as a correction
    def categorize(step, source, df):
        before = is_uncategorized(df["Category"])
        df = step(df)
        record_assignments(df.loc[before & ~is_uncategorized(df["Category"])], source)
        return df

    new_df = categorize(apply_merchant_map, "map", new_df)
    if _env_enabled("LLM_CATEGORIZE") or _config_enabled(config, "llm_enabled"):
        new_df = categorize(lambda df: apply_llm_classifier(df, history_df=sheets_history), "llm", new_df)
    min_sim = float(os.getenv("TFIDF_MIN_SIM", "0.35"))
    new_df = categorize(
        lambda df: apply_tfidf_classifier(df, min_similarity=min_sim, history_df=sheets_history),
        "tfidf",
        new_df,
    )

    # Deduplicate with existing data
    master_df = None
    if sheets_history is not None:
        master_df = sheets_history.copy()   # Pull existing data from sheets history
    # error handling for reading existing formatted file
    elif Path(out_file).exists():
        try:
            master_df = pd.read_csv(out_file)
        except pd.errors.EmptyDataError:
            master_df = None

    # Deduplicate new rows against master data
    combined, new_only = dedupe_new_rows(new_df, master_df)

    # Sort, add totals, and prepare output dataframes
    combined_sorted = sort_transactions(combined)
    combined_sorted = add_overall_total(combined_sorted)
    combined_out = combined_sorted.drop(columns=["Source", "OrigOrder"], errors="ignore")
    new_only_sorted = sort_transactions(new_only)
    new_only_sorted = attach_overall_total(new_only_sorted, combined_sorted)
    new_only_out = new_only_sorted.drop(columns=["Source", "OrigOrder"], errors="ignore")

    # Normalize date columns for both dataframes
    if "Date" in combined_out.columns:
        combined_out["Date"] = normalize_date_series(combined_out["Date"])
    if "Date" in new_only_out.columns:
        new_only_out["Date"] = normalize_date_series(new_only_out["Date"])

    # Upload to Google Sheets if sync is enabled, else save locally
    if sync_enabled:
        uploader = SheetsUploader()
        upload_mode = uploader.get_mode()
        if upload_mode == "append":
            if not new_only_out.empty:
                uploader.upload_df(new_only_out)
                print("Uploaded to Google Sheets")
            else:
                print("No new rows to append")
        else:
            uploader.upload_df(combined_out)
            print("Uploaded to Google Sheets")
    else:
        print("Merged and saved to:", out_file)

# Execute main function with error handling
if __name__ == "__main__":
    import traceback, sys
    try:
        main()
    except LedgerError as e:
        # Expected problem: message already explains the fix
        print(f"\nError: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
    except HttpError as e:
        # Google Sheets API rejected the request (bad spreadsheet_id, no access, sheet tab missing, ...)
        print(f"\nGoogle Sheets error {e.status_code}: {e.reason}")
        print("Check spreadsheet_id and sheet_name in secrets/gsheets_config.json, "
              "and that the signed-in Google account can edit the sheet.")
        input("Press Enter to exit...")
        sys.exit(1)
    except Exception:
        # Unexpected bug: show the full traceback
        traceback.print_exc()
        input("Error occurred. Press Enter to exit...")
        sys.exit(1)
