import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "create_techqa_tiny.py"
spec = importlib.util.spec_from_file_location("create_techqa_tiny", SCRIPT)
tiny = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiny)

PRODUCTS = ["websphere", "datacap", "cognos", "maximo", "netcool", "tivoli", "filenet", "domino"]


def product_of(name: str) -> str:
    return PRODUCTS[int(name[3:8]) % len(PRODUCTS)]


def doc(i: int) -> str:
    p = PRODUCTS[i % len(PRODUCTS)]
    return f"Title: {p} troubleshooting note {i}\n\nText:\n{p} error code E{i:04d} appears in the {p} server log."


@pytest.fixture
def source(tmp_path):
    """800 documents of 8 product families; all questions are about the first two families only (100 usable + 30 impossible)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(800):
        (corpus / f"swg{i:05d}.txt").write_text(doc(i), encoding="utf-8")
    rows = []
    for i in [i for i in range(400) if i % len(PRODUCTS) < 2]:
        rows.append({"id": f"Q{i}", "question": f"How do I fix error E{i:04d} in {PRODUCTS[i % len(PRODUCTS)]}?", "answer": f"restart {i}",
                     "is_impossible": False, "contexts": [{"filename": f"swg{i:05d}.txt", "text": doc(i)}]})
    for i in range(30):
        rows.append({"id": f"I{i}", "question": f"Unanswerable question number {i} about nothing", "answer": "-", "is_impossible": True,
                     "contexts": []})
    train = tmp_path / "train.json"
    train.write_text(json.dumps(rows), encoding="utf-8")
    return train, corpus


def test_default_distractors_are_random_and_spread_over_all_families(source):
    train, corpus = source
    result = tiny.build(train, corpus, seed=1, n_distractors=100)
    tiny.check(result)
    roles = [c["role"] for c in result["contexts"]]
    assert roles.count("gold") == 80 and roles.count("distractor") == 100 and "hard_negative" not in roles
    assert len({product_of(c["context_id"]) for c in result["contexts"] if c["role"] == "distractor"}) == len(PRODUCTS)


def test_zero_hard_negatives_gives_the_same_corpus_as_before(source):
    train, corpus = source
    explicit = tiny.build(train, corpus, seed=7, n_distractors=60, n_hard=0)
    default = tiny.build(train, corpus, seed=7, n_distractors=60)
    assert [c["context_id"] for c in explicit["contexts"]] == [c["context_id"] for c in default["contexts"]]


def test_hard_negatives_come_from_the_questions_product_families_and_are_never_gold(source):
    train, corpus = source
    result = tiny.build(train, corpus, seed=1, n_distractors=100, n_hard=40)
    tiny.check(result)
    gold = {c["context_id"] for c in result["contexts"] if c["role"] == "gold"}
    hard = [c for c in result["contexts"] if c["role"] == "hard_negative"]
    assert len(hard) == 40 and not gold & {c["context_id"] for c in hard}
    assert len(result["contexts"]) == 80 + 100                 # hard negatives are part of the distractor budget
    assert {product_of(c["context_id"]) for c in hard} == {"websphere", "datacap"}      # every question is about one of these two
    assert {product_of(c["context_id"]) for c in result["contexts"] if c["role"] == "distractor"} - {"websphere", "datacap"}
    again = tiny.build(train, corpus, seed=1, n_distractors=100, n_hard=40)
    assert [c["context_id"] for c in again["contexts"]] == [c["context_id"] for c in result["contexts"]]


def test_hard_negatives_are_spread_over_gold_documents(source):
    train, corpus = source
    queries = {"swg00000.txt": "websphere error E0000", "swg00001.txt": "datacap error E0001"}
    out = tiny.pick_hard_negatives(corpus, queries, set(), {"swg00000.txt", "swg00001.txt"}, 6)
    products = [product_of(n) for n, _ in out]
    assert products.count("websphere") == 3 and products.count("datacap") == 3          # round-robin, not 6 from one gold document


def test_too_many_hard_negatives_exit_with_a_message(source):
    train, corpus = source
    with pytest.raises(SystemExit):
        tiny.pick_hard_negatives(corpus, {"swg00000.txt": "websphere"}, set(), set(), 100000)
