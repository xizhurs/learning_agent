# Carbon Methodology RAG — Claude Code Instructions

## Project Purpose

This repository implements a local-first RAG system for answering questions about carbon-credit methodologies and project documents.

It is part of the broader MRV Analyst Agent project, which will eventually combine unstructured document retrieval with structured biomass, remote sensing, and geospatial analysis.

The goal is to learn and demonstrate production AI engineering through reproducible experiments, reliable evaluation, and incremental development.

**Do not rebuild functionality that already exists. Inspect the repository before proposing changes.**

## Current Implementation

The existing application includes:

- Docling document parsing, including tracked-change acceptance for DOCX.
- Configurable document chunking and embedding caching.
- LanceDB with keyword/full-text, vector, and hybrid retrieval.
- Optional cross-encoder reranking.
- Ollama embeddings and local answer generation.
- FastAPI chat interface.
- Automated retrieval evaluation and parameter ablations.
- RAGAS and Claude-assisted answer evaluation.
- Arize Phoenix tracing and latency analysis.

Configuration is managed through `config.yaml`, with environment-variable overrides.

The main Python package is `src/rag/`. Preserve the existing package structure unless a change is clearly justified.

Read `README.md`, `config.yaml`, and the relevant source modules before making architectural decisions.

## Environments

### Local

Local development is the current primary environment.

Use the existing `uv` workflow for dependencies, tests, and commands. Do not replace working libraries, package management, or infrastructure without evidence of a benefit.

### Databricks

Databricks is the target platform for later integration with structured datasets, experiment management, managed retrieval, and deployment.

Current cluster runtime: **Databricks Runtime 15.4 ML LTS**, using Python 3.11 and Spark 3.5.

Check installed package versions and enabled workspace capabilities before implementing Databricks integrations.

Keep core Python logic locally testable wherever practical.

## Engineering Principles

1. Prefer incremental improvements over architectural rewrites.
2. Preserve existing interfaces and configuration conventions.
3. Build complexity only when evaluation demonstrates a benefit.
4. Keep ingestion, retrieval, generation, evaluation, and tracing conceptually separate.
5. Use modular, typed, testable Python code.
6. Avoid unnecessary frameworks, services, and abstractions.
7. Account for latency, resource usage, reproducibility, and cloud costs.
8. Do not introduce multi-agent orchestration without a demonstrated requirement.
9. Do not silently change benchmark definitions or previously established experimental configurations.

## Retrieval and RAG

- Maintain support for keyword/full-text, vector, and hybrid retrieval.
- Keep reranking optional and configurable.
- Preserve source metadata, document identity, sections, and evidence provenance.
- Evaluate retrieval quality independently from answer generation.
- Treat better retrieval scores as insufficient evidence of better end-to-end answers.
- Require grounded answers with source references.
- Support abstention when sufficient evidence is unavailable.
- Distinguish methodology requirements from project-specific claims.

## Evaluation

Existing evaluation workflows and commands are documented in `README.md`.

Use development data for debugging and validation data for configuration selection. Reserve the held-out test set for final evaluation.

Track relevant retrieval metrics, including evidence-level hit@k, MRR, and evidence recall. Avoid relying on document-level hit@k when document titles make retrieval trivial.

Evaluate generation separately using answer correctness, faithfulness, citation support, appropriate refusal, and latency.

Treat generated question-answer labels as provisional until validated.

Do not modify reference answers, supporting quotes, or split membership merely to improve evaluation metrics.

Perform ablation studies that change one interpretable component or controlled group of components at a time.

Always inspect error cases rather than relying solely on aggregate scores.

## Existing Experimental Findings

The README documents initial QASPER retrieval and generation experiments.

Important observations:

- Cross-encoder reranking substantially improves evidence retrieval on QASPER.
- Reranking also introduces substantial CPU latency.
- At larger chunk sizes, initial retrieval mode had relatively little impact after reranking.
- Smaller chunks performed worse in the reported experiments.
- The local 3B generation model has limitations in answer correctness and abstention.
- Prompt processing and generation can dominate end-to-end latency.

These findings are dataset- and environment-specific, not universal rules.

Consult the latest evaluation artifacts before recommending changes, and validate improvements on the appropriate evaluation split.

## Tracing and Observability

Arize Phoenix is already integrated for local tracing.

Preserve existing spans for retrieval, reranking, generation, and evaluation.

Do not replace Phoenix with MLflow tracing simply because Databricks is the target platform.

Consider MLflow integration when Databricks experiment tracking or deployment provides a concrete benefit.

Capture latency, retrieval results, source references, model configuration, and errors where appropriate.

## Data Privacy

The project includes a draft project description that may contain non-public information.

Do not send private project content to hosted LLMs, embedding APIs, tracing services, or external evaluation providers without explicit authorization.

Preserve existing hosted-judge safety checks.

Do not commit credentials, sensitive documents, or generated evaluation artifacts containing restricted content to public repositories.

## Future MRV Capabilities

The broader project will eventually support:

- Read-only SQL querying over structured biomass and geospatial data.
- Python statistical analysis.
- Model performance and uncertainty comparisons.
- Agent routing between retrieval and analytical tools.
- MLflow experiment tracking and evaluation.
- Databricks integration and deployment.

Introduce these features incrementally after defining their use cases, interfaces, and evaluation criteria.

For geospatial analysis, preserve units, CRS, timestamps, model versions, and uncertainty definitions. Consider spatial dependence and blocked validation where relevant.

Do not infer significant biomass change or measurement priority from prediction values alone.

## Working Instructions for Claude

Before significant implementation:

1. Inspect relevant existing code, tests, configurations, and documentation.
2. Summarize what already exists.
3. Identify the precise missing functionality or problem.
4. Propose the smallest useful change and explain trade-offs.
5. Define acceptance criteria and tests.
6. Implement and run relevant tests.
7. Report verified results, remaining limitations, and potential regressions.

Do not claim tests passed unless executed.

Ask before destructive operations, large refactoring, dependency upgrades with compatibility risks, or provisioning paid cloud resources.

When discussing important design decisions, explain the engineering reasoning to support learning.

## Current Priorities

1. Validate and improve the carbon methodology RAG benchmark.
2. Evaluate retrieval and generation on the appropriate data splits.
3. Investigate generation correctness, abstention, and latency bottlenecks.
4. Preserve and improve local observability and reproducibility.
5. Plan the transition toward structured analytical tools and Databricks integration.

Do not implement a complete agent architecture or replace the existing local stack without a specific, evaluated reason.
