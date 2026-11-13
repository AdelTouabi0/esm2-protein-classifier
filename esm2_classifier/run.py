"""Pipeline : embeddings ESM-2 vs composition en acides aminés.

Exemple :
    python -m esm2_classifier.run data/localization.csv -o results/
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from .evaluate import cross_validate, plot_confusion, plot_projection, plot_scores
from .features import DEFAULT_MODEL, aa_composition, esm_embeddings, load_esm


def read_dataset(path):
    """Lit le CSV (accession, sequence, label) et renvoie trois listes."""
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return ([r["accession"] for r in rows], [r["sequence"] for r in rows],
            np.array([r["label"] for r in rows]))


def parse_args(argv=None):
    """Lit les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path, help="CSV accession,sequence,label")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"modèle ESM-2 Hugging Face (défaut : {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu, cuda ou mps")
    return parser.parse_args(argv)


def main(argv=None, tokenizer=None, model=None):
    """Calcule les représentations, évalue les deux modèles, écrit les figures.

    ``tokenizer`` et ``model`` peuvent être fournis directement (tests).
    Les embeddings sont mis en cache dans le dossier de sortie.
    """
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    _, sequences, labels = read_dataset(args.dataset)
    classes = sorted(set(labels))
    print(f"{len(sequences)} protéines, {len(classes)} classes : {classes}")

    cache = args.output / f"embeddings_{args.model.split('/')[-1]}.npy"
    if cache.exists():
        embeddings = np.load(cache)
    else:
        if model is None:
            tokenizer, model = load_esm(args.model)
        embeddings = esm_embeddings(sequences, tokenizer, model,
                                    batch_size=args.batch_size,
                                    device=args.device)
        np.save(cache, embeddings)
    print(f"Embeddings ESM-2 : {embeddings.shape}")

    representations = {"Composition AA": aa_composition(sequences),
                       "ESM-2": embeddings}
    scores, report = {}, {}
    for name, features in representations.items():
        predictions, fold_scores = cross_validate(features, labels)
        scores[name] = fold_scores
        report[name] = {"f1_macro_mean": round(float(fold_scores.mean()), 4),
                        "f1_macro_std": round(float(fold_scores.std()), 4)}
        print(f"\n{name} : F1 macro = {fold_scores.mean():.3f} "
              f"± {fold_scores.std():.3f}")
        print(classification_report(labels, predictions, digits=3,
                                    zero_division=0))
        slug = name.lower().replace(" ", "_").replace("-", "")
        plot_confusion(labels, predictions, classes,
                       args.output / f"confusion_{slug}.png", name)
        plot_projection(features, labels, classes,
                        args.output / f"pca_{slug}.png", f"ACP - {name}")

    plot_scores(scores, args.output / "f1_comparison.png")
    report.update({"n_proteins": len(sequences), "classes": classes,
                   "model": args.model})
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"Résultats écrits dans {args.output}/")


if __name__ == "__main__":
    main()
