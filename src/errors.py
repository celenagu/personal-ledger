# errors.py

import sys
import traceback

from googleapiclient.errors import HttpError


# Expected problems (missing files, bad config, etc.) with a message that says how to fix them.
# run() prints these without a traceback.
class LedgerError(Exception):
    pass


# Run a command's entry point with friendly error messages; pause=True waits for Enter before exiting
def run(fn, pause=False):
    try:
        fn()
        return
    except LedgerError as e:
        # Expected problem: message already explains the fix
        print(f"\nError: {e}")
    except HttpError as e:
        # Google Sheets API rejected the request (bad spreadsheet_id, no access, sheet tab missing, ...)
        print(f"\nGoogle Sheets error {e.status_code}: {e.reason}")
        print("Check spreadsheet_id and sheet_name in secrets/gsheets_config.json, "
              "and that the signed-in Google account can edit the sheet.")
    except Exception:
        # Unexpected bug: show the full traceback
        traceback.print_exc()
    if pause:
        input("Press Enter to exit...")
    sys.exit(1)
