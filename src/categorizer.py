import os
import re
from pathlib import Path
import pandas as pd

CATEGORIES_PATH = "files/rules/categories.txt"
ASSIGNMENTS_PATH = "files/rules/assigned_categories.csv"    # gitignored log of categories the script assigned
UNCATEGORIZED = "Uncategorized"

# True where a category is blank or "Uncategorized"
def is_uncategorized(series):
    categories = series.fillna("").astype(str).str.strip()
    return categories.eq("") | categories.str.lower().eq(UNCATEGORIZED.lower())

# Load the fixed list of allowed categories (one per line, # for comments); empty if no list
def load_categories(path=CATEGORIES_PATH):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    categories = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and line.lower() != UNCATEGORIZED.lower() and line not in categories:
            categories.append(line)
    return categories

_GENERIC_TOKENS = {
    "store",
    "stores",
    "payment",
    "interac",
    "canada",
    "inc",
    "ltd",
    "online",
    "pos",
    "card",
    "debit",
    "credit",
    "purchase",
    "pmt",
    "txn",
}

# Normalize merchant name for matching
def _normalize_merchant(value):
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    tokens = [tok for tok in text.split(" ") if tok and tok not in _GENERIC_TOKENS]
    return " ".join(tokens).strip()

# Load merchant mapping rules from CSV
def _load_merchant_map(map_path):
    path = Path(map_path)
    if not path.exists():
        return []

    df = pd.read_csv(path)
    if df.empty:
        return []

    df.columns = [str(c).strip().lower() for c in df.columns]
    if "merchant" not in df.columns or "category" not in df.columns:
        return []

    out = []
    for _, row in df.iterrows():
        merchant = _normalize_merchant(row.get("merchant"))
        category = str(row.get("category", "")).strip()
        if not merchant or not category:
            continue
        out.append({"merchant": merchant, "category": category})
    return out

# Apply merchant mapping rules to DataFrame
def apply_merchant_map(
    df,
    map_path="files/rules/merchant_map.csv",
    category_col="Category",
    name_col="Name",
):
    merchant_rules = _load_merchant_map(map_path)
    if not merchant_rules:
        return df          # no rules to apply, return original DataFrame

    out_df = df.copy()
    categories = out_df[category_col].fillna("").astype(str).str.strip()            # existing categories
    needs_category = categories.eq("") | categories.str.lower().eq("uncategorized") # rows needing categorization (current blank or "Uncategorized")
    normalized_names = out_df[name_col].astype(str).map(_normalize_merchant)        # normalized merchant names

    # Apply rules to rows needing categorization
    for idx in out_df.index[needs_category]:
        category = _match_rule(merchant_rules, normalized_names.at[idx])
        if category:
            out_df.at[idx, category_col] = category
    return out_df

# Find the category of the most specific (longest) rule whose merchant is a substring of the name
def _match_rule(merchant_rules, normalized_name):
    best = None
    for rule in merchant_rules:
        if rule["merchant"] in normalized_name and (best is None or len(rule["merchant"]) > len(best["merchant"])):
            best = rule
    return best["category"] if best else None

# Identify a transaction by date, merchant and amounts, so logged categories can be matched to sheet rows
def _transaction_keys(df, name_col="Name"):
    def col(name):
        return df[name] if name in df.columns else pd.Series("", index=df.index)
    def amount(name):
        values = pd.to_numeric(col(name).astype(str).str.replace(",", "").str.replace("$", ""), errors="coerce")
        return values.map(lambda v: "" if pd.isna(v) or v == 0 else f"{v:.2f}")
    date = col("Date").fillna("").astype(str).str.strip()
    name = df[name_col].astype(str).map(_normalize_merchant)
    return date + "|" + name + "|" + amount("Out") + "|" + amount("In")

# Load the log of categories the script assigned: {transaction key: (category, source)}
def _load_assignments(path=ASSIGNMENTS_PATH):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return {}
    return {k: (c, s) for k, c, s in zip(df["key"], df["category"], df["source"])}

# Log the categories this run assigned and where they came from ("map", "llm" or "tfidf"),
# so later changes in the sheet can be recognized as corrections
def record_assignments(df, source, name_col="Name", category_col="Category", path=ASSIGNMENTS_PATH):
    if df.empty:
        return
    known = _load_assignments(path)
    keys = _transaction_keys(df, name_col)
    new = {k: str(c).strip() for k, c in zip(keys, df[category_col]) if k not in known}
    if not new:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([(k, c, source) for k, c in new.items()], columns=["key", "category", "source"]).to_csv(
        p, mode="a", header=not p.exists(), index=False
    )

# Keep only history rows that are safe to learn from.
# Rows still holding a TF-IDF guess are skipped (it matches letters, not meaning);
# rows whose category was changed in the sheet since the script assigned it are marked
# "Corrected" and win over other rows for the same merchant.
def confirmed_history(history_df, name_col="Name", category_col="Category", assignments_path=ASSIGNMENTS_PATH):
    if history_df is None or history_df.empty or name_col not in history_df.columns or category_col not in history_df.columns:
        return None
    categories = history_df[category_col].fillna("").astype(str).str.strip()
    log = _load_assignments(assignments_path)
    logged = _transaction_keys(history_df, name_col).map(log)
    logged_category = logged.map(lambda v: v[0] if isinstance(v, tuple) else None)
    logged_source = logged.map(lambda v: v[1] if isinstance(v, tuple) else None)
    unchanged = logged_category.eq(categories)
    corrected = logged_category.notna() & ~unchanged
    tfidf_guess = unchanged & logged_source.eq("tfidf")

    confirmed = ~is_uncategorized(categories) & ~tfidf_guess
    out = history_df.loc[confirmed].copy()
    out[category_col] = categories[confirmed]
    out["Corrected"] = corrected[confirmed]
    return out

# Update merchant mapping rules from history DataFrame
def update_merchant_map_from_history(
    history_path="files/formatted.csv",
    map_path="files/rules/merchant_map.csv",
    name_col="Name",
    category_col="Category",
    history_df=None,
    categories_path=CATEGORIES_PATH,
):
    if history_df is None:
        history_file = Path(history_path)
        if not history_file.exists():
            return
        try:
            history_df = pd.read_csv(history_file)
        except pd.errors.EmptyDataError:
            return

    # Only learn from trusted categories (map, Gemini, or set by hand), never from TF-IDF guesses
    temp = confirmed_history(history_df, name_col, category_col)
    if temp is None or temp.empty:
        return

    # With a category list, ignore rows whose category isn't on it (e.g. old "Shopping" rows)
    allowed = load_categories(categories_path)
    if allowed:
        on_list = temp[category_col].isin(allowed)
        if not on_list.all():
            ignored = sorted(temp.loc[~on_list, category_col].unique())
            print(f"Merchant map: ignored sheet rows with categories not in {categories_path}: {', '.join(ignored)}")
        temp = temp[on_list]
    temp["merchant_norm"] = temp[name_col].map(_normalize_merchant)
    temp = temp[temp["merchant_norm"].ne("")]
    if temp.empty:
        return

    # Pick one category per merchant: corrections win over other rows, then the most common category
    counts = (
        temp.groupby(["merchant_norm", category_col])
        .agg(corrected=("Corrected", "sum"), count=("Corrected", "size"))
        .reset_index()
    )
    counts = counts.sort_values(
        ["merchant_norm", "corrected", "count", category_col],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    best = counts.drop_duplicates(subset=["merchant_norm"], keep="first")

    path = Path(map_path)
    if path.exists():
        map_df = pd.read_csv(path)
        map_df.columns = [str(c).strip().lower() for c in map_df.columns]
    else:
        map_df = pd.DataFrame(columns=["merchant", "category"])
    map_keys = map_df["merchant"].map(_normalize_merchant) if not map_df.empty else pd.Series(dtype=str)
    rules = _load_merchant_map(map_path)

    changed, added = [], []
    for _, row in best.iterrows():
        merchant, category = row["merchant_norm"], row[category_col]
        exact = map_keys.eq(merchant)
        if exact.any():
            # Existing entry for this exact merchant: update it if the user's category differs
            current = map_df.loc[exact, "category"].astype(str).str.strip()
            if not current.eq(category).all():
                map_df.loc[exact, "category"] = category
                changed.append(f"{merchant}: {current.iloc[0]} -> {category}")
        elif _match_rule(rules, merchant) != category:
            # No rule gives this category yet: add a specific entry (longest match wins)
            added.append({"merchant": merchant, "category": category})
            rules.append({"merchant": merchant, "category": category})

    if not changed and not added:
        return

    if added:
        map_df = pd.concat([map_df, pd.DataFrame(added)], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    map_df.to_csv(path, index=False)
    for line in changed:
        print(f"Merchant map updated from sheet: {line}")
    if added:
        print(f"Merchant map: learned {len(added)} new merchant(s) from sheet")

# Apply TF-IDF classifier to categorize transactions
def apply_tfidf_classifier(
    df,
    history_path="files/formatted.csv",
    min_similarity=0.35,
    category_col="Category",
    name_col="Name",
    history_df=None,
):
    # TF-IDF classification based on historical transaction names and categories
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return df

    # 
    if history_df is None:
        history_file = Path(history_path)
        if not history_file.exists():
            return df
        try:
            history_df = pd.read_csv(history_file)
        except pd.errors.EmptyDataError:
            return df
    # Train only on trusted categories, so TF-IDF doesn't learn from its own past guesses
    confirmed = confirmed_history(history_df, name_col, category_col)
    if confirmed is None or confirmed.empty:
        return df

    # Prepare training data (past transactions with known categories)
    train_names = confirmed[name_col].astype(str).map(_normalize_merchant)
    train_names_raw = confirmed[name_col].astype(str).tolist()
    train_labels = confirmed[category_col].tolist()

    # For the target df, determine which rows need categorization
    out_df = df.copy()
    categories = out_df[category_col].fillna("").astype(str).str.strip()            # existing categories
    needs_category = categories.eq("") | categories.str.lower().eq("uncategorized") # rows needing categorization
    if not needs_category.any():
        return out_df

    # for rows that need categorization, normalize the merchant names for vectorization
    target_names = out_df.loc[needs_category, name_col].astype(str).map(_normalize_merchant)
    if train_names.empty or target_names.empty:
        return out_df

    # Character n-grams are more typo-tolerant for short merchant strings
    # Configure TF-IDF vectorizer to use character n-grams (3 to 5 characters) and ignore very rare terms (min_df=1)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    train_vecs = vectorizer.fit_transform(train_names)      # 
    target_vecs = vectorizer.transform(target_names)

    # Compute cosine similarities and assign best matching categories
    # Cosine similarity: compute similarity between target & training vectors based on direction
    sims = cosine_similarity(target_vecs, train_vecs)
    best_idx = sims.argmax(axis=1)
    best_scores = sims.max(axis=1)

    # Only assign category if similarity exceeds threshold, otherwise leave as is (uncategorized)
    debug = os.getenv("TFIDF_DEBUG")
    for row_idx, label_idx, score in zip(target_names.index, best_idx, best_scores):
        if score >= min_similarity:
            out_df.at[row_idx, category_col] = train_labels[label_idx]
        if debug:
            original = out_df.at[row_idx, name_col]
            matched_name = train_names_raw[label_idx]
            matched_label = train_labels[label_idx]
            print(
                f"TFIDF: '{original}' -> '{matched_name}' "
                f"({matched_label}) score={score:.3f}"
            )

    return out_df
