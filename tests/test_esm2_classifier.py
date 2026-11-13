"""Tests hors ligne : un mini-modèle ESM aléatoire remplace ESM-2.

Le vrai modèle n'est jamais téléchargé ; on vérifie la mécanique (masques,
moyenne, ordre des séquences, pipeline) sur un modèle de même architecture.
"""

import csv
import json

import numpy as np
import pytest
import torch
from transformers import EsmConfig, EsmModel, EsmTokenizer

from esm2_classifier.features import aa_composition, esm_embeddings
from esm2_classifier.run import main

# Vocabulaire du tokenizer ESM-2 (33 jetons).
VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E",
         "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C",
         "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"]


@pytest.fixture(scope="module")
def tiny_esm(tmp_path_factory):
    """Renvoie un tokenizer ESM et un petit modèle ESM aléatoire."""
    vocab = tmp_path_factory.mktemp("esm") / "vocab.txt"
    vocab.write_text("\n".join(VOCAB))
    tokenizer = EsmTokenizer(str(vocab))
    torch.manual_seed(0)
    config = EsmConfig(vocab_size=len(VOCAB), hidden_size=16,
                       num_hidden_layers=1, num_attention_heads=2,
                       intermediate_size=32, max_position_embeddings=1100,
                       pad_token_id=1, mask_token_id=32,
                       position_embedding_type="rotary")
    model = EsmModel(config).eval()
    return tokenizer, model


def test_aa_composition_sums_to_one():
    """Chaque ligne est une distribution de fréquences."""
    comp = aa_composition(["AAAC", "WWWW"])
    assert comp.shape == (2, 20)
    assert comp.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert comp[1, "ACDEFGHIKLMNPQRSTVWY".index("W")] == 1.0


def test_embedding_is_mean_over_residues(tiny_esm):
    """L'embedding est la moyenne des résidus, sans <cls> ni <eos>."""
    tokenizer, model = tiny_esm
    seq = "MKTAYIAK"
    emb = esm_embeddings([seq], tokenizer, model)
    with torch.no_grad():
        hidden = model(**tokenizer([seq], return_tensors="pt"))
    expected = hidden.last_hidden_state[0, 1:-1].mean(dim=0).numpy()
    assert emb[0] == pytest.approx(expected, abs=1e-5)


def test_padding_and_order_do_not_change_embeddings(tiny_esm):
    """Le résultat ne dépend ni du lot ni du padding, et l'ordre est conservé."""
    tokenizer, model = tiny_esm
    sequences = ["MKTAYIAKQRQISFVKSHFSRQ", "MA", "GSHMLEDP"]
    batched = esm_embeddings(sequences, tokenizer, model, batch_size=3)
    alone = np.vstack([esm_embeddings([s], tokenizer, model)
                       for s in sequences])
    assert batched == pytest.approx(alone, abs=1e-4)


def test_pipeline_end_to_end(tiny_esm, tmp_path):
    """La pipeline produit métriques et figures sur un mini jeu de données."""
    rng = np.random.default_rng(0)
    biases = {"riche_K": "KKKKAELG", "riche_L": "LLLLAEKG", "riche_D": "DDDDAEKL"}
    dataset = tmp_path / "toy.csv"
    with open(dataset, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["accession", "sequence", "label"])
        for label, alphabet in biases.items():
            for i in range(15):
                seq = "".join(rng.choice(list(alphabet), 40))
                writer.writerow([f"{label}_{i}", seq, label])
    tokenizer, model = tiny_esm
    main([str(dataset), "-o", str(tmp_path / "out")],
         tokenizer=tokenizer, model=model)
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["n_proteins"] == 45
    # La composition suffit à séparer ces classes construites pour l'être.
    assert metrics["Composition AA"]["f1_macro_mean"] > 0.9
    assert (tmp_path / "out" / "f1_comparison.png").exists()
