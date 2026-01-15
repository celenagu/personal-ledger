# personal-ledger
Easier on the eyes.

Script to automate personal transaction tracking. Processes CSV file account transaction exports from bank (optimized for TD), categorizes transactions, and uploads them to a centralized Google Sheet. 

### Features
* Cleans up and standardizes transaction files from different accounts (chequing, savings, credit)
* Assigns categories to transactions using a merchant map, learns from historical data.
    * Merchant-to-category mappings can be defined in `rules/merchant_mapping.csv`
    * Mappings updated in Google Sheet swill be reflected in merchant map
    * Uses TF-IDF classifer to assign categories for new transactions based on similarity to past ones
* Keeps a master sheet updated

Files currently under `files` and `secrets` are sample files for demo purposes. They do not contain any real information.

### Set up
1. This project uses several Python packages. These can be installed using the provided `requirements.txt` file. 
    ```bash
    pip install -r requirements.txt
    ```
2. Set up Google Sheets API credentials in `secrets/gsheets_credentials.json`
3. Configure spreadsheet details in `secrets/gsheets_config.json`
4. Place input CSVs (`chequing.csv`, `savings.csv`, `credit.csv`) in the `files/input` directory

### Run
```bash
python src/main.py
```
