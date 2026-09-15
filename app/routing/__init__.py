# app/routing package
# Phase 2 — Intelligent Routing
#
# Part 1 — Request Analysis & Feature Extraction
# Import analyzer directly: from app.routing.analyzer import ...
#
# Part 3 — Model Capability & Candidate Selection
# Import selector directly: from app.routing.candidate_selector import ...
#
# Deliberately NOT re-exporting all symbols here to avoid circular imports:
# candidate_selector imports from app.classification, which imports from
# app.routing.analyzer, creating a cycle if __init__.py re-exports both.
