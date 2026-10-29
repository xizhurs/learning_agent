from rag.eval.gen_testset import _quote_in_passages
from rag.eval.metrics import first_hit_rank, is_refusal, quote_coverage, retrieval_summary
from rag.generate import NOT_FOUND

QUOTE = "The match results are deemed valid where SDM scores for each covariate are at less than or equal to 0.25."
GOOD = "Header > Section\n\n" + QUOTE + " The final selection of control plots is then fixed."
BAD = "Projects must demonstrate additionality using an approved performance benchmark."


def test_quote_coverage_distinguishes_support():
    assert quote_coverage(QUOTE, GOOD) == 1.0
    assert quote_coverage(QUOTE, BAD) < 0.2


def test_quote_coverage_spans_chunks_only_via_best_segment():
    two_part = (
        "First supporting sentence about leakage belts. ... Second sentence about SDM scores below 0.25 for covariates."
    )
    chunk = "Second sentence about SDM scores below 0.25 for covariates."
    assert quote_coverage(two_part, chunk) == 1.0


def test_first_hit_rank_and_summary():
    assert first_hit_rank(QUOTE, [BAD, GOOD, BAD], 0.6) == 2
    assert first_hit_rank(QUOTE, [BAD, BAD], 0.6) is None
    s = retrieval_summary([1, 2, None, 5], [1, 3, 5])
    assert s["hit@1"] == 0.25 and s["hit@3"] == 0.5 and s["hit@5"] == 0.75
    assert abs(s["mrr"] - (1 + 0.5 + 0.2) / 4) < 1e-9


def test_refusal_detection():
    assert is_refusal(NOT_FOUND)
    assert not is_refusal("The threshold is 0.25 [1].")


def test_import_quote_check_is_markdown_tolerant_and_catches_fabrication():
    passage = "| **Data unit** | Dimensionless |\n| **Description** | Number of plots in each cohort (i). |"
    assert _quote_in_passages("Number of plots in each cohort (i).", [passage])
    assert not _quote_in_passages("Number of hectares in each project zone is estimated annually.", [passage])
