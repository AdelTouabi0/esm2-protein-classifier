# Localisation subcellulaire des protéines avec ESM-2

![tests](https://github.com/AdelTouabi0/esm2-protein-classifier/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Transformers-orange)
![Licence MIT](https://img.shields.io/badge/licence-MIT-green)

> *Protein language model embeddings (ESM-2) vs. amino-acid composition for
> predicting subcellular localization, with a leakage-aware evaluation.*

Peut-on prédire **où se trouve une protéine dans la cellule** (noyau,
cytoplasme, mitochondrie, membrane plasmique, milieu extracellulaire) à
partir de sa seule séquence ? Ce projet compare deux représentations :

| Représentation | Dimension | Ce qu'elle capture |
|---|---|---|
| Composition en acides aminés | 20 | fréquences des résidus, sans leur ordre |
| Embedding **ESM-2** | 320 | contexte de chaque résidu appris par un modèle de langage protéique |

Les deux sont évaluées avec **le même classifieur** (régression logistique),
pour que la différence de performance reflète la représentation et non le
modèle.

## Méthode

1. **Données** (`scripts/build_dataset.py`) : protéines humaines revues de
   Swiss-Prot, interrogées via l'API REST d'UniProt. Seules les protéines
   annotées dans **une seule** des cinq localisations sont gardées, puis les
   classes sont équilibrées (400 protéines par classe par défaut).
2. **Embeddings** : le modèle `esm2_t6_8M_UR50D` (8 millions de paramètres,
   utilisable sur un ordinateur portable) produit un vecteur par résidu ; on
   en fait la moyenne sur les seuls résidus, en excluant les jetons spéciaux
   et le remplissage grâce au masque d'attention.
3. **Évaluation** : validation croisée stratifiée à 5 plis, F1 macro (même
   poids pour chaque classe), matrices de confusion et projection ACP.

## Installation et utilisation

```bash
git clone https://github.com/AdelTouabi0/esm2-protein-classifier.git
cd esm2-protein-classifier
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/build_dataset.py -o data/localization.csv
python -m esm2_classifier.run data/localization.csv -o results/
```

Le modèle ESM-2 est téléchargé automatiquement depuis Hugging Face au
premier lancement. Les embeddings sont mis en cache dans `results/`.
Pour un modèle plus grand : `--model facebook/esm2_t12_35M_UR50D`
(et `--device cuda` sur GPU).

## Résultats

*Section à compléter après exécution.* La pipeline produit :
`f1_comparison.png` (F1 macro des deux représentations), les matrices de
confusion et les projections ACP de chaque représentation, et
`metrics.json`.

## Limites et pistes

- **Fuite par homologie** : des protéines très similaires peuvent se
  retrouver à la fois dans l'entraînement et dans le test, ce qui gonfle les
  scores. La correction standard consiste à regrouper les séquences par
  identité (par exemple avec MMseqs2 à 30 %) et à découper par groupes
  (`GroupKFold`). C'est la prochaine étape prévue.
- Les protéines multi-localisées, écartées ici, demanderaient une
  classification multi-étiquette.
- La moyenne sur les résidus perd l'information de position (un peptide
  signal N-terminal est un indice fort de sécrétion) : un pooling par
  attention ou un fine-tuning d'ESM-2 sont des extensions naturelles.

## Tests

```bash
pytest -q
```

Les tests n'ont besoin d'aucun téléchargement : un petit modèle ESM aléatoire
de même architecture remplace ESM-2. Ils vérifient notamment que
l'embedding est bien la moyenne des seuls résidus et qu'il ne dépend ni du
padding ni de la composition des lots.

## Structure

```
esm2_classifier/
├── features.py   # composition en acides aminés, embeddings ESM-2
├── evaluate.py   # validation croisée, figures
└── run.py        # pipeline complète
scripts/build_dataset.py   # jeu de données UniProt
tests/
```

## Références

- Lin Z. *et al.* (2023). Evolutionary-scale prediction of atomic-level
  protein structure with a language model. *Science*, 379, 1123-1130.
- Almagro Armenteros J. J. *et al.* (2017). DeepLoc: prediction of protein
  subcellular localization using deep learning. *Bioinformatics*, 33,
  3387-3395.

## Contexte

Projet personnel, Master 1 Bio-informatique. Développé avec l'assistance
d'outils d'IA (Claude).

## Auteur

Adel Touabi, Master 1 Bio-informatique, Université Paris Cité.
