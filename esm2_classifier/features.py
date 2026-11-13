"""Représentations des protéines : embeddings ESM-2 et composition en acides aminés.

ESM-2 (Lin et al., 2023) est un modèle de langage protéique de type
Transformer, entraîné à prédire des acides aminés masqués sur des millions
de séquences. Ses représentations internes capturent des propriétés
structurales et fonctionnelles, sans aucune annotation.
"""

import numpy as np
import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"


def aa_composition(sequences):
    """Renvoie la fréquence des 20 acides aminés de chaque séquence (n, 20).

    C'est la représentation de référence, la plus simple possible : elle
    ignore complètement l'ordre des résidus.
    """
    features = np.zeros((len(sequences), len(AMINO_ACIDS)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(AMINO_ACIDS):
            features[i, j] = seq.count(aa)
        total = features[i].sum()
        if total:
            features[i] /= total
    return features


def load_esm(model_name=DEFAULT_MODEL):
    """Charge le tokenizer et le modèle ESM-2 depuis Hugging Face."""
    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def esm_embeddings(sequences, tokenizer, model, batch_size=8,
                   max_length=1022, device="cpu"):
    """Renvoie un embedding par protéine (moyenne sur les résidus).

    Les représentations de la dernière couche sont moyennées sur les seuls
    résidus : les jetons spéciaux (<cls>, <eos>) et le remplissage (padding)
    sont exclus grâce au masque d'attention.
    """
    model = model.to(device)
    outputs = []
    # Tri par longueur : les lots contiennent des séquences de taille
    # proche, ce qui réduit le padding et accélère le calcul.
    order = np.argsort([len(s) for s in sequences])
    for start in range(0, len(sequences), batch_size):
        batch = [sequences[i][:max_length]
                 for i in order[start:start + batch_size]]
        tokens = tokenizer(batch, return_tensors="pt", padding=True)
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].clone()
        mask[:, 0] = 0                          # jeton <cls>
        lengths = tokens["attention_mask"].sum(dim=1)
        mask[torch.arange(len(batch)), lengths - 1] = 0   # jeton <eos>
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        outputs.append(pooled.cpu().numpy())
    embeddings = np.concatenate(outputs)
    # Remise dans l'ordre d'origine des séquences.
    result = np.empty_like(embeddings)
    result[order] = embeddings
    return result
