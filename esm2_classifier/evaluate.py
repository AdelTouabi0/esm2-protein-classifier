"""Comparaison de classifieurs par validation croisée stratifiée."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")


def make_classifier(seed=0):
    """Régression logistique précédée d'une standardisation des variables."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, C=0.5,
                                            random_state=seed))


def cross_validate(features, labels, n_splits=5, seed=0):
    """Renvoie les prédictions hors-pli et le F1 macro de chaque pli.

    Chaque protéine est prédite par un modèle qui ne l'a jamais vue. Le F1
    macro donne le même poids à chaque classe.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions = cross_val_predict(make_classifier(seed), features, labels,
                                    cv=cv)
    fold_scores = [f1_score(labels[test], predictions[test], average="macro")
                   for _, test in cv.split(features, labels)]
    return predictions, np.array(fold_scores)


def plot_confusion(labels, predictions, classes, output_path, title):
    """Trace la matrice de confusion normalisée par ligne."""
    matrix = confusion_matrix(labels, predictions, labels=classes)
    norm = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="Proportion")
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Localisation prédite")
    ax.set_ylabel("Localisation réelle")
    for (i, j), value in np.ndenumerate(norm):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                color="white" if value > 0.5 else "black", fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_projection(features, labels, classes, output_path, title):
    """Projette les représentations en 2D (ACP) colorées par classe."""
    coords = PCA(n_components=2, random_state=0).fit_transform(
        StandardScaler().fit_transform(features))
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in classes:
        mask = labels == name
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6,
                   label=name)
    ax.set_xlabel("CP1")
    ax.set_ylabel("CP2")
    ax.legend(markerscale=2, fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scores(scores, output_path):
    """Diagramme en barres du F1 macro (moyenne et écart-type sur les plis)."""
    names = list(scores)
    means = [scores[n].mean() for n in names]
    stds = [scores[n].std() for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, means, yerr=stds, capsize=6, color=["#9aa5b1", "#3b6ea8"])
    ax.set_ylabel("F1 macro (validation croisée 5 plis)")
    ax.set_ylim(0, 1)
    for i, mean in enumerate(means):
        ax.text(i, mean + 0.03, f"{mean:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
