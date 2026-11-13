"""Construit un jeu de données de localisation subcellulaire depuis UniProt.

Usage :
    python scripts/build_dataset.py -o data/localization.csv

Protéines humaines revues (Swiss-Prot) annotées dans exactement une des cinq
localisations ci-dessous. Les protéines multi-localisées sont écartées pour
garder une tâche de classification à une seule étiquette.
"""

import argparse
import csv
import io
import random
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://rest.uniprot.org/uniprotkb/stream"
# Termes du vocabulaire contrôlé UniProt des localisations subcellulaires.
LOCATIONS = {
    "Nucleus": "SL-0191",
    "Cytoplasm": "SL-0086",
    "Mitochondrion": "SL-0173",
    "Secreted": "SL-0243",
    "Cell membrane": "SL-0039",
}


def fetch(term, max_length):
    """Renvoie {accession: séquence} pour une localisation donnée."""
    query = (f"(reviewed:true) AND (organism_id:9606) AND (cc_scl_term:{term})"
             f" AND (length:[50 TO {max_length}])")
    url = API + "?" + urllib.parse.urlencode(
        {"query": query, "format": "tsv", "fields": "accession,sequence"})
    with urllib.request.urlopen(url, timeout=120) as response:
        text = response.read().decode("utf-8")
    rows = csv.DictReader(io.StringIO(text), delimiter="\t")
    return {row["Entry"]: row["Sequence"] for row in rows}


def main():
    """Télécharge, retire les protéines multi-localisées, équilibre et écrit."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("data/localization.csv"))
    parser.add_argument("--per-class", type=int, default=400)
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    by_location = {}
    for name, term in LOCATIONS.items():
        by_location[name] = fetch(term, args.max_length)
        print(f"{name:<14} {len(by_location[name]):>6} protéines")

    # Une accession présente dans plusieurs localisations est écartée.
    seen = {}
    for name, proteins in by_location.items():
        for accession in proteins:
            seen[accession] = seen.get(accession, 0) + 1

    rng = random.Random(args.seed)
    rows = []
    for name, proteins in by_location.items():
        unique = sorted(a for a in proteins if seen[a] == 1)
        chosen = rng.sample(unique, min(args.per_class, len(unique)))
        rows.extend((a, proteins[a], name) for a in chosen)
        print(f"{name:<14} {len(unique):>6} uniques, {len(chosen)} retenues")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["accession", "sequence", "label"])
        writer.writerows(rows)
    print(f"{len(rows)} protéines écrites dans {args.output}")


if __name__ == "__main__":
    main()
