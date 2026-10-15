# personal-ledger
Easier on the eyes.

Script to automate personal transaction tracking. Processes CSV file account transaction exports from bank (optimized for TD), categorizes transactions, and uploads them to a centralized Google Sheet. 

### Features
* Cleans up and standardizes transaction files from different accounts (chequing, savings, credit)
* Assigns categories to transactions using a merchant map, learns from historical data.
    * Merchant-to-category mappings can be defined in `rules/merchant_mapping.csv`
    * Categories you set or correct in the Google Sheet are learned into the merchant map (see "How categories are learned" below)
    * Optionally uses Google Gemini to categorize new merchants the merchant map doesn't cover (see below)
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

### How categories are learned
Each run logs the categories it assigns, and where they came from, in `files/rules/assigned_categories.csv` (gitignored).
At the start of the next run, the merchant map learns from your Google Sheet automatically; nothing needs reviewing:
* Categories from the merchant map or Gemini are learned as rules, so those merchants don't need Gemini again.
* TF-IDF guesses are never learned (it matches similar-looking names, not meaning).
* If you change any category in the sheet, it's treated as a correction and the merchant map is updated to match.
* Only categories listed in `files/rules/categories.txt` are learned; sheet rows with other categories are ignored.

### Recategorize existing sheet rows
To apply your current merchant map (and Gemini, if enabled) to rows already in the Google Sheet:
```bash
python src/recategorize.py
```
It lists every proposed change and only writes after you type `y`. Only the Category column is changed,
and the whole sheet is backed up to `files/backups/` (gitignored) first.
Run it after changing the merchant map or `categories.txt`, so old rows don't teach the map outdated categories.

### LLM categorization (optional)
Transactions the merchant map doesn't match can be categorized by Google Gemini (free tier) before falling back to TF-IDF.
Only deduplicated merchant names are sent, never dates, amounts or balances.
Note: on the free tier, Google may use what you send to improve its products.
Gemini can only assign categories listed in `files/rules/categories.txt` (one per line), or leave a transaction "Uncategorized".
Your merchant map and sheet history are sent along as examples.

1. Create a free API key at https://aistudio.google.com/apikey and paste it into `secrets/gemini_api_key.txt` (gitignored), or set `GEMINI_API_KEY`
2. Enable it with `"llm_enabled": true` in `secrets/gsheets_config.json`, or `LLM_CATEGORIZE=1` for a single run
3. Optional: `LLM_MODEL` to change the model (default `gemini-3.8-flash`), `LLM_DEBUG=1` to print each assignment

If the API call fails for any reason (no key, quota used up, network), the run continues and those transactions go to TF-IDF instead.
