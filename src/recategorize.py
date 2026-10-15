# recategorize.py
# Re-apply the current merchant map (and Gemini, if enabled) to rows already in the Google Sheet.
# Shows every change and asks before writing; only the Category column is changed.

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from googleapiclient.errors import HttpError

from categorizer import (
    apply_merchant_map,
    is_uncategorized,
    load_categories,
    record_assignments,
)
from errors import LedgerError
from formatter import canonical_column, normalize_history_df
from llm_categorizer import apply_llm_classifier
from main import PROJECT_ROOT, _config_enabled, _env_enabled, _read_gsheets_config
from sheets_uploader import SheetsUploader

BACKUP_DIR = "files/backups"    # gitignored


def recategorize():
    os.chdir(PROJECT_ROOT)
    uploader = SheetsUploader()
    raw = uploader.read_sheet()
    if raw.empty:
        print("Sheet is empty, nothing to do")
        return

    canonical = [canonical_column(c) for c in raw.columns]
    if "Category" not in canonical or "Name" not in canonical:
        raise LedgerError("Sheet needs Category and Name columns to recategorize")
    category_index = canonical.index("Category")

    sheet = normalize_history_df(raw.copy()).reset_index(drop=True)
    current = sheet["Category"].fillna("").astype(str).str.strip()
    allowed = set(load_categories())

    # What the merchant map says now, for every row
    proposed = sheet.copy()
    proposed["Category"] = "Uncategorized"
    proposed = apply_merchant_map(proposed)
    by_map = ~is_uncategorized(proposed["Category"])
    source = pd.Series("", index=sheet.index)
    source[by_map] = "map"

    # Rows the map can't place and that are uncategorized or off the category list go to Gemini
    off_list = ~current.isin(allowed) if allowed else pd.Series(False, index=sheet.index)
    need_llm = ~by_map & (is_uncategorized(current) | off_list)
    config = _read_gsheets_config()
    if need_llm.any() and (_env_enabled("LLM_CATEGORIZE") or _config_enabled(config, "llm_enabled")):
        llm = apply_llm_classifier(proposed.loc[need_llm], history_df=sheet)
        by_llm = ~is_uncategorized(llm["Category"])
        proposed.loc[llm.index[by_llm], "Category"] = llm.loc[by_llm, "Category"]
        source[llm.index[by_llm]] = "llm"
        if not by_llm.any():
            print("Gemini didn't categorize anything this time (see above); the changes below are from the merchant map only.")
            print("You can apply them now and run this again later for Gemini.\n")
    elif need_llm.any():
        print(f"Gemini is off, so {int(need_llm.sum())} row(s) the merchant map can't place are left as is "
              "(set \"llm_enabled\": true in secrets/gsheets_config.json or LLM_CATEGORIZE=1 to use it).\n")

    new = proposed["Category"].where(source.ne(""), current)
    changed = new.ne(current)

    unresolved = off_list & ~changed & ~is_uncategorized(current)
    if unresolved.any():
        left = current[unresolved].value_counts()
        print(f"{int(unresolved.sum())} row(s) have categories not on the list and nothing could replace them (left as is):")
        for cat, n in left.items():
            print(f"  {cat}: {n}")
        print()

    if not changed.any():
        print("No changes: every row already matches the merchant map")
        return

    summary = (
        pd.DataFrame({"Name": sheet["Name"], "From": current, "To": new, "By": source})[changed]
        .groupby(["Name", "From", "To", "By"]).size().rename("Rows").reset_index()
        .sort_values(["Rows", "Name"], ascending=[False, True])
    )
    print(f"Proposed changes ({int(changed.sum())} rows):")
    for _, r in summary.iterrows():
        print(f"  {r['Name']}: {r['From'] or '(blank)'} -> {r['To']}  [{r['By']}, {r['Rows']} row(s)]")
    print()

    try:
        answer = input(f"Apply {int(changed.sum())} change(s) to the Google Sheet? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer != "y":
        print("Nothing changed")
        return

    # Back up the whole sheet locally before writing
    Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)
    backup = Path(BACKUP_DIR) / f"sheet_backup_{datetime.now():%Y%m%d_%H%M%S}.csv"
    raw.to_csv(backup, index=False)
    print(f"Backed up sheet to {backup}")

    # Write only the Category column; keep untouched cells exactly as they were
    column = raw.iloc[:, category_index].where(~changed.values, new.values)
    uploader.write_column(category_index, ["" if pd.isna(v) else v for v in column])

    # Log the new categories so later edits in the sheet are recognized as corrections
    updated = sheet[changed].copy()
    updated["Category"] = new[changed]
    for src in ("map", "llm"):
        record_assignments(updated[source[changed].eq(src)], src)
    print(f"Updated {int(changed.sum())} row(s)")


if __name__ == "__main__":
    try:
        recategorize()
    except LedgerError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except HttpError as e:
        print(f"\nGoogle Sheets error {e.status_code}: {e.reason}")
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
