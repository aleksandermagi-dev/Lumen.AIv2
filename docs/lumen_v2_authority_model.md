# Lumen V2 Authority Model Notes

This document is not the current runtime authority.

For implementation-truth boundaries, use [runtime_architecture_authority.md](C:\Users\aleks\Desktop\lumen1.1\docs\runtime_architecture_authority.md).

This note captures authority policy and deferred cleanup work beyond the current runtime architecture.

## Goal

Unify understanding and authority so fewer layers recompute overlapping interpretations of the same prompt.

## Current v1.5 bridge

- `DomainRouter` remains the route authority.
- route-support helpers can inform downstream shaping, but cannot re-select route.
- memory retrieval can influence reply content and continuity, but cannot choose mode.
- explanation shaping can rewrite the visible answer surface, but cannot reclassify intent.

## V2 targets

1. Unify prompt understanding across:
   - NLU extraction
   - route-support signals
   - domain routing
   - downstream explanatory/continuation consumers
2. Reduce duplicated interpretation work for:
   - explanatory prompts
   - continuation confidence
   - project-return / thread-return signals
3. Narrow orchestration concentration in `InteractionService.ask(...)` without changing behavior contracts.

## Explicitly not part of the current pass

- no routing rewrite
- no memory redesign
- no lane-enforcement redesign
- no merger of conversation, planning, and research reply models
