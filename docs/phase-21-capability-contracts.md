# Phase 21: Safe Capability Expansion and No-Self-Edit Lock

> Historical note: this document records an earlier implementation/audit phase and is not the current runtime authority. For current status, see [README.md](../README.md), [LUMEN_V2_RELEASE_STATUS.md](../LUMEN_V2_RELEASE_STATUS.md), and [LUMEN_V2_ARCHITECTURE.md](../LUMEN_V2_ARCHITECTURE.md).

## Summary
Phase 21 tightens Lumen's capability honesty and only expands adjacent workflows that the current runtime can actually support.

Implemented focus areas:
- no-self-edit lock
- writing/editing capability honesty
- bounded dataset/license guidance
- bounded invention/design status surfacing
- compact capability transparency in diagnostics and the desktop presenter

## Capability Contract
Supported:
- explainability / transparency

Bounded:
- dataset and local analysis workflows
- invention and design support

Provider-gated:
- writing and editing workflows that require hosted generation

Not promised:
- self-editing / self-modification
- autonomous RPA / broad automation
- speech / audio understanding
- vision / image understanding
- live news and politics authority
- investing advice
- health-risk prediction / diagnosis / treatment
- fabrication-grade schematic signoff

## Runtime Behavior
- Self-edit prompts are refused deterministically.
- Dataset-license and source prompts return bounded advisory guidance.
- Hosted writing/editing requests are surfaced honestly:
  - if a hosted provider is configured, Lumen can perform the task
  - if not, Lumen reports the workflow as provider-gated instead of pretending it is locally available
- Design and invention outputs are marked as bounded rather than implied to be fabrication-ready.

## Transparency Surfaces
- Doctor diagnostics now include a capability-contract summary.
- Desktop decorated replies can surface compact capability-status lines for bounded, provider-gated, and not-promised domains.
- Phase 20 audit language now reflects that self-editing is disabled, not merely supervised.
