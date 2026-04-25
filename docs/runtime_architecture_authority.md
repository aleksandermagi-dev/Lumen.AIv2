# Runtime Architecture Authority

This document is the current runtime truth for Lumen in this repository.

Use this file as the implementation authority when code and architecture notes differ.

## Runtime authority rules

1. `PromptNLU` is the only builder of the canonical prompt-understanding contract.
2. The Language Structure Layer is a sub-layer of prompt understanding, not a parallel authority.
3. `DomainRouter` is the only route selector.
4. Memory and retrieval are advisory only. They may shape content and continuity, but they must not choose route or mode.
5. Tool execution is explicit capability execution, not passive retrieval.
6. The response pipeline owns interactive turn packaging.
7. Reporting owns durable, exportable, archive-facing, and CLI/report formatting.

## Canonical prompt-understanding contract

The canonical contract is built in [src/lumen/nlu/prompt_nlu.py](C:\Users\aleks\Desktop\lumen1.1\src\lumen\nlu\prompt_nlu.py) and modeled in [src/lumen/nlu/models.py](C:\Users\aleks\Desktop\lumen1.1\src\lumen\nlu\models.py).

It includes:

- original prompt
- normalized prompt
- canonical reconstructed prompt
- shared surface views
- structure interpretation
- language
- topic
- intent
- entities
- advisory profile hints

Downstream layers should consume this contract or a read-only projection of it instead of rebuilding prompt interpretation locally.

## Authority boundaries

### Routing

- Owner: [src/lumen/routing/domain_router.py](C:\Users\aleks\Desktop\lumen1.1\src\lumen\routing\domain_router.py)
- Input: canonical prompt-understanding or router view
- Output: route authority decision
- Not allowed:
  - memory-selected route changes
  - response-shaping route changes
  - tool-layer route overrides

### Retrieval

- Owner: [src/lumen/reasoning/memory_retrieval_layer.py](C:\Users\aleks\Desktop\lumen1.1\src\lumen\reasoning\memory_retrieval_layer.py)
- Input: chosen route plus canonical prompt context
- Output: advisory retrieval context only
- Not allowed:
  - changing selected route
  - changing selected response mode
  - implicitly invoking tools

### Tool execution

- Owner: [src/lumen/services/tool_execution_service.py](C:\Users\aleks\Desktop\lumen1.1\src\lumen\services\tool_execution_service.py)
- Input: explicit execution request
- Output: tool result plus archive writeback
- Not allowed:
  - passive retrieval behavior
  - silent route selection

### Response pipeline

- Owner: [src/lumen/reasoning/response_packaging_support.py](C:\Users\aleks\Desktop\lumen1.1\src\lumen\reasoning\response_packaging_support.py)
- Scope: interactive reply shaping, packaging, observability
- Not allowed:
  - changing route authority
  - acting as reporting/export layer

### Reporting

- Owner: [src/lumen/reporting/output_formatter.py](C:\Users\aleks\Desktop\lumen1.1\src\lumen\reporting\output_formatter.py)
- Scope: durable/exportable/report-facing payload formatting
- Not allowed:
  - interactive route decisions
  - turn-level reply composition authority

## Layer ownership matrix

### Conversational

- Owns: interaction moves, pacing, ask-vs-answer shape, thread continuity cues
- Does not own: route selection, safety classification, retrieval authority

### Cognitive

- Owns: reasoning posture, pattern framing, internal thought organization
- Does not own: final route selection, packaging authority

### Support/Stability

- Owns: retries, weak-output detection, clarification gating, recovery behavior
- Does not own: primary meaning interpretation, route authority

### Tool

- Owns: explicit external capability invocation
- Does not own: passive knowledge retrieval, route selection

### Retrieval

- Owns: memory and archive context selection
- Does not own: route selection, tool execution

### Reporting

- Owns: durable/report/export formatting
- Does not own: live turn response composition

## Document relationship

- [docs/runtime_architecture_authority.md](C:\Users\aleks\Desktop\lumen1.1\docs\runtime_architecture_authority.md): current runtime truth
- [docs/lumen_v2_authority_model.md](C:\Users\aleks\Desktop\lumen1.1\docs\lumen_v2_authority_model.md): authority-policy and deferred-cleanup note
- [LUMEN_V2_ARCHITECTURE.md](C:\Users\aleks\Desktop\lumen1.1\LUMEN_V2_ARCHITECTURE.md): high-level current architecture map
