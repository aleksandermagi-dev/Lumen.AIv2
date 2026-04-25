# Knowledge Seed Hygiene

Use this checklist before adding or expanding local seed entries.

## Entry checklist

- title is specific, stable, and human-readable
- aliases are specific enough to avoid broad collisions
- summary explains the concept cleanly without depending on internal jargon
- related links are meaningful, not just numerous
- comparison relationships are intentional and symmetric when appropriate

## Alias guardrails

- avoid very short aliases unless they are unusually unambiguous
- avoid aliases that are common cross-domain abbreviations
- prefer full phrases over generic fragments
- if an alias would plausibly refer to multiple common subjects, do not add it

## Coverage guardrails

- common-concept additions should have exact or near-exact lookup coverage
- near-neighbor concepts should get explicit regression tests
- new comparisons should include reversed-comparison tests
- generic prompts that should still miss should keep missing after seed growth
