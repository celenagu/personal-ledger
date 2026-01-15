import os
import re
from pathlib import Path
import pandas as pd


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


def apply_merchant_map(
    df,
    map_path="files/rules/merchant_map.csv",
    category_col="Category",
    name_col="Name",
):
    merchant_rules = _load_merchant_map(map_path)
    if not merchant_rules:
        return df

    out_df = df.copy()
    categories = out_df[category_col].fillna("").astype(str).str.strip()
    needs_category = categories.eq("") | categories.str.lower().eq("uncategorized")
    normalized_names = out_df[name_col].astype(str).map(_normalize_merchant)

    for idx in out_df.index[needs_category]:
        name = normalized_names.at[idx]
        for rule in merchant_rules:
            if rule["merchant"] in name:
                out_df.at[idx, category_col] = rule["category"]
                break
    return out_df


def update_merchant_map_from_history(
    history_path="files/formatted.csv",
    map_path="files/rules/merchant_map.csv",
    name_col="Name",
    category_col="Category",
    history_df=None,
):
    if history_df is None:
        history_file = Path(history_path)
        if not history_file.exists():
            return
        try:
            history_df = pd.read_csv(history_file)
        except pd.errors.EmptyDataError:
            return
    if history_df.empty or name_col not in history_df.columns or category_col not in history_df.columns:
        return

    categories = history_df[category_col].fillna("").astype(str).str.strip()
    valid = ~categories.str.lower().eq("uncategorized") & categories.ne("")
    if not valid.any():
        return

    base_map = _load_merchant_map(map_path)
    base_merchants = {entry["merchant"] for entry in base_map}

    temp = history_df.loc[valid, [name_col, category_col]].copy()
    temp["merchant_norm"] = temp[name_col].map(_normalize_merchant)
    temp["category_norm"] = temp[category_col].astype(str).str.strip()
    temp = temp[temp["merchant_norm"].ne("")]

    if temp.empty:
        return

    counts = (
        temp.groupby(["merchant_norm", "category_norm"])
        .size()
        .rename("count")
        .reset_index()
    )
    counts = counts.sort_values(
        ["merchant_norm", "count", "category_norm"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    best = counts.drop_duplicates(subset=["merchant_norm"], keep="first")

    updates = []
    def _overlaps_existing(existing, merchant):
        for known in existing:
            if known in merchant or merchant in known:
                return True
        return False
    for _, row in best.iterrows():
        merchant = row["merchant_norm"]
        if merchant in base_merchants:
            continue
        if _overlaps_existing(base_merchants, merchant):
            continue
        updates.append({"merchant": merchant, "category": row["category_norm"]})
        base_merchants.add(merchant)

    if not updates:
        return

    path = Path(map_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(updates)
    if path.exists():
        existing = pd.read_csv(path)
        out_df = pd.concat([existing, out_df], ignore_index=True)
    out_df.to_csv(path, index=False)


def apply_tfidf_classifier(
    df,
    history_path="files/formatted.csv",
    min_similarity=0.35,
    category_col="Category",
    name_col="Name",
    history_df=None,
):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return df

    if history_df is None:
        history_file = Path(history_path)
        if not history_file.exists():
            return df
        try:
            history_df = pd.read_csv(history_file)
        except pd.errors.EmptyDataError:
            return df
    if history_df.empty or name_col not in history_df.columns or category_col not in history_df.columns:
        return df

    history_categories = history_df[category_col].fillna("").astype(str).str.strip()
    history_valid = ~history_categories.str.lower().eq("uncategorized") & history_categories.ne("")
    if not history_valid.any():
        return df

    train_names = history_df.loc[history_valid, name_col].astype(str).map(_normalize_merchant)
    train_names_raw = history_df.loc[history_valid, name_col].astype(str).tolist()
    train_labels = history_df.loc[history_valid, category_col].astype(str).str.strip().tolist()

    out_df = df.copy()
    categories = out_df[category_col].fillna("").astype(str).str.strip()
    needs_category = categories.eq("") | categories.str.lower().eq("uncategorized")
    if not needs_category.any():
        return out_df

    target_names = out_df.loc[needs_category, name_col].astype(str).map(_normalize_merchant)
    if train_names.empty or target_names.empty:
        return out_df

    # Character n-grams are more typo-tolerant for short merchant strings.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    train_vecs = vectorizer.fit_transform(train_names)
    target_vecs = vectorizer.transform(target_names)

    sims = cosine_similarity(target_vecs, train_vecs)
    best_idx = sims.argmax(axis=1)
    best_scores = sims.max(axis=1)

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
