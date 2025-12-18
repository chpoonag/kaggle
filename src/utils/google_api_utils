import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class GoogleSheetHandler:
    def __init__(self,
                 service_account_info: dict=None,
                 service_account_info_path: str=None,
                 spreadsheet_id: str=None,
                 scopes=[
                    'https://www.googleapis.com/oauth/spreadsheets',
                    # 'https://www.googleapis.com/auth/realtime-bidding',
                ]):
        assert not((service_account_info, service_account_info_path)==(None, None)),\
            f"Expect at least service_account_info or service_account_info_path to be provided."
        assert (service_account_info is None or service_account_info_path is None), \
            f"Only one of service_account_info or service_account_info_path should be provided."
        if service_account_info_path:
            with open(service_account_info_path, 'r') as file:
                service_account_info = json.load(file)
        self.service_account_info = service_account_info
        self.scopes = scopes
        self.spreadsheet_id = spreadsheet_id
        self.service = self._authenticate()

    def _authenticate(self):
        """Authenticates with Google Sheets API using service account info."""
        try:
            credentials = service_account.Credentials.from_service_account_info(
                self.service_account_info, scopes=self.scopes
            )
            service = build('sheets', 'v4', credentials=credentials)
            print("Successfully authenticated with Google Sheets API.")
            return service
        except HttpError as err:
            print(f"Authentication error: {err}")
            return None

    def read_sheet(self, range_name, spreadsheet_id=None):
        """Reads data from a specified range in a Google Sheet."""
        spreadsheet_id =  self.spreadsheet_id if spreadsheet_id is None else spreadsheet_id
        if not self.service:
            print("Service not initialized. Cannot read sheet.")
            return []
        try:
            sheet = self.service.spreadsheets()
            result = sheet.values().get(
                spreadsheetId=spreadsheet_id,
                range=range_name
            ).execute()
            values = result.get('values', [])
            if not values:
                print('No data found.')
            else:
                print(f"Data from spreadsheet '{spreadsheet_id}' range '{range_name}':")
                for row in values:
                    print(row)
            return values
        except HttpError as err:
            print(f"Error reading sheet: {err}")
            return []

    def update_sheet_values(self, range_name, values, spreadsheet_id=None):
        """
        Updates a range of values in a Google Sheet.

        Args:
            spreadsheet_id: The ID of the spreadsheet.
            range_name: The A1 notation of the range to update (e.g., 'Sheet1!A1:B2').
            values: A list of lists representing the rows and columns to write.
        """
        spreadsheet_id = self.spreadsheet_id if spreadsheet_id is None else spreadsheet_id
        if not self.service:
            print("Service not initialized. Cannot update sheet.")
            return None
        try:
            body = {
                'values': values
            }
            result = self.service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption='USER_ENTERED', # Renders input as if a user typed it
                body=body
            ).execute()
            print(f"{result.get('updatedCells')} cells updated in '{spreadsheet_id}' range '{range_name}'.")
            return result
        except HttpError as err:
            print(f"Error updating sheet: {err}")
            return None

    def append_rows(self, values, spreadsheet_id=None, range_name='Sheet1', offset_rows=0, offset_cols=0):
        """
        Appends new rows of data to the end of a Google Sheet with optional offsets.

        Args:
            values: A list of lists representing the rows and columns to append.
            spreadsheet_id: The ID of the spreadsheet. If None, uses the initialized spreadsheet_id.
            range_name: The sheet name to append to (e.g., 'Sheet1').
            offset_rows: Number of empty rows to insert before the actual data.
            offset_cols: Number of empty columns to prepend to each row of data.
        """
        spreadsheet_id = self.spreadsheet_id if spreadsheet_id is None else spreadsheet_id
        if not self.service:
            print("Service not initialized. Cannot append rows.")
            return None
        try:
            final_values = []

            # 1. Apply column offset to existing data rows
            if values:
                for row in values:
                    final_values.append([''] * offset_cols + row)
            
            # 2. Determine the width for empty rows (if any)
            # If there are already rows (after column offset), use their maximum width.
            # Otherwise, if column offset was specified, use that as the width.
            # Otherwise, default to a width of 1 for empty rows.
            max_width = 0
            if final_values:
                max_width = max(len(row) for row in final_values)
            elif offset_cols > 0:
                max_width = offset_cols
            else:
                max_width = 1 # Minimum width for empty rows if no data and no col offset

            # 3. Prepend empty rows if offset_rows > 0
            if offset_rows > 0:
                empty_rows_to_prepend = [[''] * max_width for _ in range(offset_rows)]
                final_values = empty_rows_to_prepend + final_values

            # If after all processing, final_values is empty, it means nothing to append
            if not final_values:
                print("No data or meaningful offsets provided to append.")
                return None

            body = {
                'values': final_values
            }
            result = self.service.spreadsheets().values().append(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption='USER_ENTERED',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()
            print(f"{result.get('updates').get('updatedCells')} cells appended to '{spreadsheet_id}' in '{range_name}'.")
            return result
        except HttpError as err:
            print(f"Error appending rows: {err}")
            return None
