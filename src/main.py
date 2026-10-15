# main.py

# Import necessary modules
from pathlib import Path
import os
import json
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
from errors import run

# All paths (files/, secrets/) are relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==============================
# Helper Functions
# ==============================

# Read secrets/gsheets_config.json ({} if missing or invalid)
def _read_gsheets_config(path="secrets/gsheets_config.json"):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}

# Gemini runs when LLM_CATEGORIZE=1 or "llm_enabled": true in the config
def llm_enabled(config):
    value = config.get("llm_enabled")
    if isinstance(value, str):
        value = value.strip().lower() in {"1", "true", "yes", "on"}
    return os.getenv("LLM_CATEGORIZE") == "1" or value is True or value == 1

# ==============================
# Main Functionality
# ==============================

# main function to process and upload data
def main():
    # Run from the project root so relative paths work no matter where the script is launched from
    os.chdir(PROJECT_ROOT)

    # Merge input files
    new_df = merge_input_files()        # df = pd.DataFrame

    # Note: what is a data frame?
    # 2D, size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns).

    # Read what's already in the Google Sheet and learn from it
    config = _read_gsheets_config()
    sheets_history = normalize_history_df(SheetsUploader().read_sheet())
    if sheets_history.empty:
        sheets_history = None
    update_merchant_map_from_history(history_df=sheets_history)

    # Only categorize rows that aren't already in the sheet (saves Gemini quota)
    _, new_df = dedupe_new_rows(new_df, sheets_history)
    print(f"{len(new_df)} new transaction(s) to categorize")

    # apply merchant map, then LLM (if enabled), then TF-IDF for anything still uncategorized
    # Each step's results are logged with their source: TF-IDF guesses are never learned as rules,
    # and any category later changed in the sheet is recognized as a correction
    def categorize(step, source, df):
        if df.empty:
            return df
        before = is_uncategorized(df["Category"])
        df = step(df)
        record_assignments(df.loc[before & ~is_uncategorized(df["Category"])], source)
        return df

    new_df = categorize(apply_merchant_map, "map", new_df)
    if llm_enabled(config):
        new_df = categorize(lambda df: apply_llm_classifier(df, history_df=sheets_history), "llm", new_df)
    min_sim = float(os.getenv("TFIDF_MIN_SIM", "0.35"))
    new_df = categorize(
        lambda df: apply_tfidf_classifier(df, min_similarity=min_sim, history_df=sheets_history),
        "tfidf",
        new_df,
    )

    # Combine the categorized new rows with the existing data
    combined, new_only = dedupe_new_rows(new_df, sheets_history)

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

    # Upload to Google Sheets: only new rows in append mode, everything in overwrite mode
    uploader = SheetsUploader()
    if uploader.get_mode() == "append":
        if not new_only_out.empty:
            uploader.upload_df(new_only_out)
            print("Uploaded to Google Sheets")
        else:
            print("No new rows to append")
    else:
        uploader.upload_df(combined_out)
        print("Uploaded to Google Sheets")

# Execute main function with error handling
if __name__ == "__main__":
    run(main, pause=True)
