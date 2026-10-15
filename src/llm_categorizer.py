# llm_categorizer.py

import os
from pathlib import Path
from typing import List

import pandas as pd
from pydantic import BaseModel

from categorizer import CATEGORIES_PATH, UNCATEGORIZED, _load_merchant_map, confirmed_history, load_categories

DEFAULT_MODEL = "gemini-3.8-flash"     # Gemini API free tier
API_KEY_PATH = "secrets/gemini_api_key.txt"


# GEMINI_API_KEY env var; otherwise read the key from the secrets file
def _api_key(path=API_KEY_PATH):
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return Path(path).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None
MAX_EXAMPLES = 300      # cap on labeled examples sent as context


# Parsed response: one category per merchant name
class _Assignment(BaseModel):
    name: str
    category: str

class _Assignments(BaseModel):
    assignments: List[_Assignment]


# Response schema sent to Gemini: category restricted to the allowed list
def _response_schema(categories):
    return {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "category": {"type": "string", "enum": categories + [UNCATEGORIZED]},
                    },
                    "required": ["name", "category"],
                },
            },
        },
        "required": ["assignments"],
    }


# Collect labeled examples from the merchant map and confirmed sheet history, keeping only allowed categories
def _examples(map_path, history_df, name_col, category_col, allowed):
    examples = {}
    for rule in _load_merchant_map(map_path):
        examples[rule["merchant"]] = rule["category"]

    confirmed = confirmed_history(history_df, name_col, category_col)
    if confirmed is not None:
        for name, cat in zip(confirmed[name_col].astype(str), confirmed[category_col]):
            examples.setdefault(name.strip(), cat)

    allowed_set = set(allowed)
    return [(name, cat) for name, cat in examples.items() if cat in allowed_set][:MAX_EXAMPLES]


def _build_prompt(names, categories, examples):
    example_lines = "\n".join(f"{name} -> {cat}" for name, cat in examples)
    name_lines = "\n".join(names)
    return (
        "Categorize these Canadian bank/credit card transactions by merchant name.\n\n"
        f"Allowed categories (use one exactly as written):\n{chr(10).join(categories)}\n\n"
        f"How I've categorized past transactions:\n{example_lines}\n\n"
        f"If none of the categories clearly fits, use \"{UNCATEGORIZED}\".\n"
        "Return one assignment per merchant name below, with the name copied exactly.\n\n"
        f"Merchant names:\n{name_lines}"
    )


# Categorize remaining uncategorized rows with Gemini; returns df unchanged on any failure
def apply_llm_classifier(
    df,
    map_path="files/rules/merchant_map.csv",
    categories_path=CATEGORIES_PATH,
    category_col="Category",
    name_col="Name",
    history_df=None,
):
    out_df = df.copy()
    categories = out_df[category_col].fillna("").astype(str).str.strip()
    needs_category = categories.eq("") | categories.str.lower().eq(UNCATEGORIZED.lower())
    if not needs_category.any():
        return out_df

    allowed = load_categories(categories_path)
    if not allowed:
        print(f"LLM: no categories listed in {categories_path}, skipping")
        return out_df
    examples = _examples(map_path, history_df, name_col, category_col, allowed)

    try:
        import httpx
        from google import genai
        from google.genai import errors
    except ImportError:
        print("LLM: google-genai package not installed (pip install -r requirements.txt), skipping")
        return out_df

    # Only merchant names are sent, deduplicated; no dates, amounts or balances
    names = sorted(set(out_df.loc[needs_category, name_col].astype(str).str.strip()))
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)

    api_key = _api_key()
    if not api_key:
        print(f"LLM: no Gemini API key found (put it in {API_KEY_PATH} or set GEMINI_API_KEY), skipping")
        return out_df

    try:
        # Retry when Gemini is busy (429/500/502/503/504): up to 5 tries, waiting 2s, 4s, 8s... (max 30s)
        client = genai.Client(
            api_key=api_key,
            http_options={"retry_options": {"attempts": 5, "initial_delay": 2, "max_delay": 30}},
        )
        response = client.models.generate_content(
            model=model,
            contents=_build_prompt(names, allowed, examples),
            config={
                "response_mime_type": "application/json",
                "response_json_schema": _response_schema(allowed),
            },
        )
    except errors.ClientError as e:
        # 400 bad key, 403 no access, 404 unknown model, 429 free-tier quota used up
        print(f"LLM: Gemini API rejected the request ({e.code} {e.status}): {e.message}, skipping")
        return out_df
    except errors.ServerError as e:
        print(f"LLM: Gemini API server error ({e.code}) after 5 tries, skipping (try again later)")
        return out_df
    except httpx.TransportError:
        print("LLM: could not reach Gemini API (network?), skipping")
        return out_df

    try:
        parsed = _Assignments.model_validate_json(response.text or "")
    except ValueError:
        print("LLM: no usable response from Gemini (blocked or malformed), skipping")
        return out_df

    # Double-check answers against the list; anything else stays uncategorized
    allowed_set = set(allowed)
    lookup = {
        a.name.strip(): a.category.strip()
        for a in parsed.assignments
        if a.category.strip() in allowed_set
    }

    debug = os.getenv("LLM_DEBUG")
    assigned = 0
    for idx in out_df.index[needs_category]:
        name = str(out_df.at[idx, name_col]).strip()
        category = lookup.get(name)
        if category:
            out_df.at[idx, category_col] = category
            assigned += 1
        if debug:
            print(f"LLM: '{name}' -> {category or UNCATEGORIZED}")

    print(f"LLM: categorized {assigned} of {int(needs_category.sum())} transactions")
    return out_df
