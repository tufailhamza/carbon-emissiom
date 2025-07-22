import json
import pandas as pd
from rapidfuzz import fuzz, process


def load_pluto_data(json_path="nyc_pluto_metadata.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def fuzzy_match_address(input_address, pluto_df, top_n=1):
    # Make all addresses uppercase
    input_address = input_address.upper()

    # Extract address column from the dataframe
    addresses = pluto_df['address'].astype(str).str.upper().tolist()

    # Use rapidfuzz to get top matches
    matches = process.extract(input_address, addresses, scorer=fuzz.token_sort_ratio, limit=top_n)

    # Return the best matching row(s)
    matched_rows = pluto_df[pluto_df['address'].str.upper().isin([match[0] for match in matches])]
    return matched_rows.assign(match_score=[m[1] for m in matches])

# Usage
pluto_df = load_pluto_data()
user_input = "650 77th Street"  # slightly different from "650 77 STREET"
matched = fuzzy_match_address(user_input, pluto_df)

print("Matched Addresses:")
print(matched)
