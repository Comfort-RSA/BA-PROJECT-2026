"""
build_schools_json.py

Expands schools.json to cover every South African secondary school by name,
using the free official DBE EMIS "Schools Masterlist" export - while being
honest that we only KNOW which streams are offered for the schools you've
curated by hand. For every other school, "streams" is left empty and the
website will say "not verified" instead of guessing.

Where to get the source file(s):
    https://www.lurits2.doe.gov.za/HomeLinks/Masterlist
    Download the Schools Masterlist for each province you want (or all of
    them). They come as .xlsx files with one row per school. Put the
    file(s) in this folder - one or many, this script reads every .xlsx
    and .csv file in EMIS_INPUT_DIR.

Column names in the real DBE export can vary slightly by year/province, so
this script matches common variants (case-insensitive, spaces/underscores
ignored) instead of expecting exact column names. If it can't find a needed
column, it will tell you which one and stop, so you can check the file.

What it does:
    1. Reads every school row from the EMIS file(s).
    2. Keeps only Secondary and Combined-phase schools (Grade 10-12 exists
       there; Primary-only schools are dropped).
    3. For each school, checks if it matches (by name) a school already in
       your CURRENT_SCHOOLS_PATH (your hand-curated schools.json). If it
       does, keeps that school's real "streams" and "subjects" data.
       If not, adds it with "streams": [] (meaning: not verified).
    4. Writes the combined result to OUTPUT_PATH.

Run:
    python build_schools_json.py
"""

import json
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
EMIS_INPUT_DIR = BASE_DIR               # put the downloaded EMIS file(s) here
CURRENT_SCHOOLS_PATH = BASE_DIR / "schools.json"   # your hand-curated file
OUTPUT_PATH = BASE_DIR / "schools_full.json"       # review this, then rename to schools.json

# Column name variants the DBE export has used. Matching is case-insensitive
# and ignores spaces/underscores, so "Institution Name", "institution_name"
# and "INSTITUTIONNAME" all match the same entry here.
COLUMN_ALIASES = {
    "name": ["institutionname", "schoolname", "name", "institution"],
    "province": ["province", "provincename"],
    "municipality": ["districtmunicipality", "municipality", "district", "circuit", "circuitname"],
    "phase": ["phase", "phaseofeducation", "phasename", "institutiontype"],
}
# Substrings to look for in the phase/InstitutionType column (case-insensitive).
# The LURITS export uses InstitutionType values like "Secondary School",
# "Combined School", etc. rather than a plain "Secondary"/"Combined" phase
# column, so we match on "contains" instead of an exact value.
KEEP_PHASE_SUBSTRINGS = ["secondary", "combined"]


def _norm(s):
    return re.sub(r"[\s_]+", "", str(s).strip().lower())


def _find_column(columns, aliases):
    lookup = {_norm(c): c for c in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def load_emis_files():
    files = sorted(list(EMIS_INPUT_DIR.glob("*.xlsx")) + list(EMIS_INPUT_DIR.glob("*.csv")))
    files = [f for f in files if f.name not in {"schools.json", "schools_full.json"}]
    if not files:
        raise SystemExit(
            f"No .xlsx or .csv files found in {EMIS_INPUT_DIR}.\n"
            "Download the DBE Schools Masterlist for your province(s) from\n"
            "https://www.lurits2.doe.gov.za/HomeLinks/Masterlist\n"
            "and put the file(s) in this folder, then run this script again."
        )

    frames = []
    for f in files:
        df = pd.read_csv(f) if f.suffix == ".csv" else pd.read_excel(f)

        col_map = {}
        for field, aliases in COLUMN_ALIASES.items():
            found = _find_column(df.columns, aliases)
            if found is None and field != "municipality":  # municipality is optional
                raise SystemExit(
                    f"Couldn't find a '{field}' column in {f.name}.\n"
                    f"Columns found: {list(df.columns)}\n"
                    f"Add the real column name to COLUMN_ALIASES['{field}'] in this script."
                )
            if found:
                col_map[field] = found

        clean = pd.DataFrame({field: df[col] for field, col in col_map.items()})
        frames.append(clean)
        print(f"Read {len(clean)} rows from {f.name}")

    return pd.concat(frames, ignore_index=True)


def main():
    emis = load_emis_files()

    if "phase" in emis.columns:
        before = len(emis)
        phase_lower = emis["phase"].astype(str).str.strip().str.lower()
        pattern = "|".join(KEEP_PHASE_SUBSTRINGS)
        emis = emis[phase_lower.str.contains(pattern, regex=True, na=False)]
        print(f"Kept {len(emis)}/{before} rows with phase containing any of {KEEP_PHASE_SUBSTRINGS}")

    emis = emis.dropna(subset=["name"]).drop_duplicates(subset=["name"])
    print(f"{len(emis)} unique secondary/combined schools after cleaning")

    with open(CURRENT_SCHOOLS_PATH, encoding="utf-8") as f:
        curated = json.load(f)["schools"]
    curated_by_name = {s["name"].strip().lower(): s for s in curated}
    print(f"{len(curated_by_name)} hand-curated schools loaded from {CURRENT_SCHOOLS_PATH.name}")

    combined = []
    matched = 0
    for _, row in emis.iterrows():
        name = str(row["name"]).strip()
        existing = curated_by_name.get(name.lower())
        if existing:
            combined.append(existing)   # keep the real, hand-checked streams/subjects
            matched += 1
        else:
            combined.append({
                "name": name,
                "province": str(row.get("province", "")).strip(),
                "municipality": str(row.get("municipality", "")).strip(),
                "subjects": [],
                "streams": [],   # not verified - the website should say so, not guess
            })

    # Any hand-curated school that wasn't in the EMIS file (e.g. a private
    # school, or a name that didn't match exactly) is kept too.
    combined_names = {s["name"].strip().lower() for s in combined}
    for s in curated:
        if s["name"].strip().lower() not in combined_names:
            combined.append(s)

    print(f"Matched {matched} EMIS schools to your curated stream data")
    print(f"Total schools in output: {len(combined)} "
          f"({matched} with verified streams, {len(combined) - matched} not verified)")

    output = {
        "note": (
            f"{len(combined)} South African secondary/combined schools, sourced from the "
            "DBE EMIS Schools Masterlist (names/province/municipality). 'streams' is only "
            "filled in for schools that have been manually verified - for every other "
            "school it is an empty list, meaning 'not verified', not 'offers nothing'."
        ),
        "schools": combined,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUTPUT_PATH.name}. Check it, then rename it to schools.json to use it.")


if __name__ == "__main__":
    main()