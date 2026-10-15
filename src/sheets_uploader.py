import json
from pathlib import Path

import pandas as pd
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from errors import LedgerError
from formatter import canonical_column

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def _sheet_range(sheet_name, cell_range=None):
    safe_name = str(sheet_name).replace("'", "''")
    prefix = f"'{safe_name}'"
    if not cell_range:
        return prefix
    return f"{prefix}!{cell_range}"


# 0-based column index to A1 letters (0 -> A, 26 -> AA)
def _column_letter(index):
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _load_config(config_path):
    path = Path(config_path)
    if not path.exists():
        raise LedgerError(f"Missing config file: {config_path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as e:
        raise LedgerError(f"{config_path} is not valid JSON: {e}")


def _sign_in(credentials_path):
    if not Path(credentials_path).exists():
        raise LedgerError(
            f"Missing {credentials_path}. Download an OAuth client JSON from Google Cloud Console "
            "(APIs & Services > Credentials) and save it there."
        )
    try:
        flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
    except ValueError as e:
        raise LedgerError(f"{credentials_path} is not a valid OAuth client file: {e}")
    print("Opening browser to sign in to Google...")
    return flow.run_local_server(port=0)


def _load_credentials(credentials_path, token_path):
    creds = None
    token_file = Path(token_path)
    if token_file.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        except ValueError:
            # Corrupt or incomplete token file: ignore it and sign in again
            creds = None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                # Token was revoked or expired (e.g. app in "Testing" mode, or unused for months)
                print("Saved Google sign-in has expired.")
                creds = _sign_in(credentials_path)
        else:
            creds = _sign_in(credentials_path)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json(), encoding="utf-8")
    return creds


def _get_sheet_id(service, spreadsheet_id, sheet_name):
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for sheet in spreadsheet.get("sheets", []):
        props = sheet.get("properties", {})
        if props.get("title") == sheet_name:
            return props.get("sheetId")
    return None


def _set_date_format(service, spreadsheet_id, sheet_id, column_index):
    if sheet_id is None or column_index is None:
        return
    body = {
        "requests": [
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startColumnIndex": column_index,
                        "endColumnIndex": column_index + 1,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "numberFormat": {
                                "type": "DATE",
                                "pattern": "yyyy-mm-dd",
                            }
                        }
                    },
                    "fields": "userEnteredFormat.numberFormat",
                }
            }
        ]
    }
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=body,
    ).execute()


class SheetsUploader:
    def __init__(
        self,
        config_path="secrets/gsheets_config.json",
        credentials_path="secrets/gsheets_credentials.json",
        token_path="secrets/gsheets_token.json",
    ):
        self.config_path = config_path
        self.credentials_path = credentials_path
        self.token_path = token_path

    def read_sheet(self):
        config = _load_config(self.config_path)
        spreadsheet_id = config.get("spreadsheet_id")
        sheet_name = config.get("sheet_name")

        if not spreadsheet_id or not sheet_name:
            raise LedgerError("secrets/gsheets_config.json must include spreadsheet_id and sheet_name")

        creds = _load_credentials(self.credentials_path, self.token_path)
        service = build("sheets", "v4", credentials=creds)

        response = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=_sheet_range(sheet_name),
        ).execute()
        values = response.get("values", [])
        if not values:
            return pd.DataFrame()

        headers = values[0]
        rows = values[1:]
        return pd.DataFrame(rows, columns=headers)

    def get_mode(self):
        config = _load_config(self.config_path)
        return config.get("mode", "append").lower()

    # Overwrite one column's data cells (row 2 down) with the given values, leaving other columns untouched
    def write_column(self, column_index, values):
        config = _load_config(self.config_path)
        spreadsheet_id = config.get("spreadsheet_id")
        sheet_name = config.get("sheet_name")
        if not spreadsheet_id or not sheet_name:
            raise LedgerError("secrets/gsheets_config.json must include spreadsheet_id and sheet_name")

        creds = _load_credentials(self.credentials_path, self.token_path)
        service = build("sheets", "v4", credentials=creds)
        letter = _column_letter(column_index)
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=_sheet_range(sheet_name, f"{letter}2:{letter}{len(values) + 1}"),
            valueInputOption="RAW",
            body={"values": [[v] for v in values]},
        ).execute()

# upload DataFrame to Google Sheets with append or overwrite mode
    def upload_df(self, df):
        config = _load_config(self.config_path)
        spreadsheet_id = config.get("spreadsheet_id")
        sheet_name = config.get("sheet_name")
        mode = config.get("mode", "append").lower()

        if not spreadsheet_id or not sheet_name:
            raise LedgerError("secrets/gsheets_config.json must include spreadsheet_id and sheet_name")

        creds = _load_credentials(self.credentials_path, self.token_path)
        service = build("sheets", "v4", credentials=creds)

        preferred = ["Date", "Category", "Name", "Type", "Out", "In", "Total"]
        if all(col in df.columns for col in preferred):
            ordered = preferred + [col for col in df.columns if col not in preferred]
            df = df[ordered]

        if "Date" in df.columns:
            df = df.copy()
            df["Date"] = df["Date"].astype(str)

        values = [df.columns.tolist()]
        for row in df.itertuples(index=False):
            values.append(["" if pd.isna(cell) else cell for cell in row])

        if mode == "overwrite":
            service.spreadsheets().values().clear(
                spreadsheetId=spreadsheet_id,
                range=_sheet_range(sheet_name),
            ).execute()
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=_sheet_range(sheet_name, "A1"),
                valueInputOption="USER_ENTERED",
                body={"values": values},
            ).execute()
            sheet_id = _get_sheet_id(service, spreadsheet_id, sheet_name)
            if "Date" in df.columns:
                _set_date_format(service, spreadsheet_id, sheet_id, df.columns.get_loc("Date"))
            return

        if mode != "append":
            raise LedgerError("mode in secrets/gsheets_config.json must be 'append' or 'overwrite'")

        existing = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=_sheet_range(sheet_name, "1:1"),
        ).execute()
        header = (existing.get("values") or [[]])[0]
        if not header:
            payload = values
        else:
            # Line the new rows up with the sheet's existing columns,
            # adding any new columns to the end of the header row
            canonical = [canonical_column(h) for h in header]
            extra = [c for c in df.columns if c not in canonical]
            if extra:
                service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=_sheet_range(sheet_name, "A1"),
                    valueInputOption="USER_ENTERED",
                    body={"values": [header + extra]},
                ).execute()
                canonical = canonical + extra
            df = df.reindex(columns=canonical)
            payload = [["" if pd.isna(cell) else cell for cell in row] for row in df.itertuples(index=False)]
        if not payload:
            return
        service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=_sheet_range(sheet_name),
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": payload},
        ).execute()
        sheet_id = _get_sheet_id(service, spreadsheet_id, sheet_name)
        if "Date" in df.columns:
            _set_date_format(service, spreadsheet_id, sheet_id, df.columns.get_loc("Date"))

    def upload_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        self.upload_df(df)
