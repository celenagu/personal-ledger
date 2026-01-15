# main.py

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
    update_merchant_map_from_history,
)
from sheets_uploader import SheetsUploader


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


def _truthy(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _config_enabled(config, key):
    return _truthy(config.get(key))


def _env_enabled(name):
    return os.getenv(name) == "1"

def main():
    out_file = "files/formatted.csv"

    new_df = merge_input_files()


    config = _read_gsheets_config()
    sync_enabled = _env_enabled("SHEETS_SYNC") or _config_enabled(config, "sync_enabled")
    history_enabled = _env_enabled("SHEETS_HISTORY") or _config_enabled(config, "history_enabled") or sync_enabled
    sheets_history = None
    if history_enabled:
        sheets_history = SheetsUploader().read_sheet()
        sheets_history = normalize_history_df(sheets_history)
        if sheets_history is not None and sheets_history.empty:
            sheets_history = None

    if sheets_history is None:
        update_merchant_map_from_history()
    else:
        update_merchant_map_from_history(history_df=sheets_history)

    new_df = apply_merchant_map(new_df)
    min_sim = float(os.getenv("TFIDF_MIN_SIM", "0.35"))
    new_df = apply_tfidf_classifier(
        new_df,
        min_similarity=min_sim,
        history_df=sheets_history,
    )

    # Append-only master with duplicate handling
    master_df = None
    if sheets_history is not None:
        master_df = sheets_history.copy()
    elif Path(out_file).exists():
        try:
            master_df = pd.read_csv(out_file)
        except pd.errors.EmptyDataError:
            master_df = None

    combined, new_only = dedupe_new_rows(new_df, master_df)

    combined_sorted = sort_transactions(combined)
    combined_sorted = add_overall_total(combined_sorted)
    combined_out = combined_sorted.drop(columns=["Source", "OrigOrder"], errors="ignore")
    new_only_sorted = sort_transactions(new_only)
    new_only_sorted = attach_overall_total(new_only_sorted, combined_sorted)
    new_only_out = new_only_sorted.drop(columns=["Source", "OrigOrder"], errors="ignore")
    if "Date" in combined_out.columns:
        combined_out["Date"] = normalize_date_series(combined_out["Date"])
    if "Date" in new_only_out.columns:
        new_only_out["Date"] = normalize_date_series(new_only_out["Date"])
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


if __name__ == "__main__":
    import traceback, sys
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("Error occurred. Press Enter to exit...")
        sys.exit(1)
