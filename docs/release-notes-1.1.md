# Lumen 1.1 Overview

This note keeps the longer `1.0 -> 1.1` summary out of the main README while making the current state of the repo easy to audit.

## From 1.0 to 1.1

`1.0` established the local-first Lumen baseline:

- manifest-backed tool discovery
- the early core bundle surface
- session and archive scaffolding
- the first desktop chat loop
- initial routing, math, system, knowledge, memory, and reporting behavior

`1.1` expands that baseline into a broader integrated runtime.

## Major Additions in 1.1

### v2.1 Foundation Slice

The first `v2.1` major-lift slice is now present in source and extends the `1.1` runtime with shared analysis and research bundles:

- `data.describe`
- `data.correlate`
- `data.regression`
- `data.cluster`
- `data.visualize`
- `viz.graph`
- `viz.network`
- `viz.timeline`
- `viz.parameter_space`
- `paper.search`
- `paper.summary`
- `paper.compare`
- `paper.extract.methods`
- `physics.energy_model`
- `simulate.system`
- `simulate.orbit`
- `simulate.population`
- `simulate.diffusion`
- `astronomy.orbit_profile`
- `experiment.design`
- `experiment.variables`
- `experiment.controls`
- `experiment.analysis_plan`
- `invent.generate_concepts`
- `invent.constraint_check`
- `invent.material_suggestions`
- `invent.failure_modes`

These bundles plug into the existing reasoning spine instead of bypassing it, so tool results can be integrated into final responses while preserving the current `collab`, `default`, and `direct` mode contract.

### Conversational / NLG Layer

- A dedicated mode-response shaping layer now sits after reasoning and tool execution.
- It keeps `collab`, `default`, and `direct` more consistent across:
  - follow-up offers
  - tool missing-input surfaces
  - runtime/failure surfaces
- This layer changes delivery posture without changing route choice, facts, or conclusions.

### Thin Domain Wrappers

- The first thin wrapper slice is now present with:
  - `physics.energy_model`
  - `astronomy.orbit_profile`
- These wrappers sit on top of the shared helpers instead of creating new isolated vertical systems.
- The goal is domain-facing interpretation and prompt reachability, not replacing the shared core bundles underneath them.

### Knowledge and Retrieval

- Local knowledge coverage was expanded substantially across astronomy, physics, history, biology, chemistry, earth science, engineering, systems, and math support.
- A glossary-backed retrieval layer now helps normalize aliases and canonical concepts without turning into a second competing knowledge store.
- Known-topic prompts now prefer direct grounded answers more reliably, while unknown topics still refuse safely.

### Clarification and Conversation

- Clarification behavior was refined rather than removed.
- Follow-up handling improved for `yes`, `no`, `keep current route`, and pivot-style replies.
- Mode behavior was tightened so `default`, `collab`, and `direct` keep the same facts while varying presentation.
- Comparison, depth, and relational explanation prompts now behave more consistently on grounded knowledge topics.

### ANH Runtime

- ANH is now an active `1.1` bundle surface.
- The current ANH path supports:
  - direct raw FITS input
  - processed result/summary handling
  - raw MAST zip/archive intake with explicit staging before science execution
- Runtime diagnostics were improved so dependency, extraction, and artifact-generation failures are easier to distinguish.

### Content Integration

- Content capabilities route through Lumen’s tool system instead of existing as an external identity.
- Missing provider/config cases now return clearer runtime diagnostics.
- Formatter and generation failures are safer and more explicit than the earlier raw exception paths.

### Data, Visualization, and Paper Workflows

- Structured data inputs can now be analyzed from attached `CSV`, `TSV`, and `JSON` files.
- Data workflows currently support description, correlation, regression, clustering, and visualization.
- Visualization workflows can generate graph, network, timeline, and parameter-space artifacts with structured metadata.
- Paper workflows can summarize supplied paper text, compare papers, extract methods, and return explicit runtime diagnostics when no paper-search source is configured.
- Generic attached-file prompts now have better routing for structured data and plain-text paper content, instead of only ANH spectral inputs.

### Simulation Workflows

- A new `simulate` bundle now supports bounded local simulations for:
  - first-order systems
  - orbit geometry
  - logistic population growth
  - one-dimensional diffusion
- These simulations are parameter-driven and artifact-backed, rather than arbitrary code execution.
- Simulation outputs are routed back through the same reasoning/response path as the other live bundles.

### Experiment Workflows

- A new `experiment` bundle now supports bounded local planning for:
  - experiment design
  - variables
  - controls
  - analysis plans
- These workflows stay descriptive and structured rather than turning into open-ended lab protocol generation.
- Experiment outputs are routed back through the same reasoning/response path as the other live bundles.

### Invent Workflows

- A new `invent` bundle now supports bounded local planning for:
  - concept generation
  - constraint checks
  - material suggestions
  - failure-mode analysis
- These workflows stay high-level and constraint-aware rather than turning into unconstrained invention prompts.
- Invent outputs are routed back through the same reasoning/response path as the other live bundles.

### Desktop UX

- The desktop app now supports file, folder, and zip attachment for the next send.
- Attachments are passed directly into the controller as `input_path`.
- A confirmation-first stop control returns the UI to the user immediately and ignores late worker results.
- Starter prompts were updated to reflect the current knowledge/tool surface and are now surfaced as visible prompt chips.
- The shell now uses a darker, cleaner visual treatment with less border stacking and clearer panel hierarchy.
- `Memory`, `Last Chats`, and `Settings` are built lazily so the main chat workspace appears faster on startup.

### Validation Status

- Lumen completed a full v2 end-to-end validation pass across source runtime and packaged desktop smoke coverage.
- Current runtime truth:
  - `ANH`: real execution verified
  - `content`: runtime/provider gated until a dedicated hosted setup is configured for Lumen
- The current advertised desktop and tool surface is aligned with that validation posture.

## Current Known Limitations

- The local knowledge base is curated, not encyclopedic.
- Some tool and content capabilities still depend on external runtime/provider configuration.
- Paper search needs a configured paper source before live search can complete.
- Simulation is intentionally bounded and simplified; it is a foundation layer for later domain wrappers, not a high-fidelity physics engine.
- Experiment planning is intentionally bounded and high-level; it is meant for structured planning support, not operational wet-lab or hazardous procedure guidance.
- Invent workflows are intentionally bounded and conceptual; they do not replace detailed engineering validation or safety review.
- Natural-language coverage is broader than in `1.0`, but some less-common tool phrasings can still under-route.
- Packaged runtime behavior should still be smoke-tested after rebuild whenever source changes land close to QA.

## QA Notes

Useful high-signal checks for the current `1.1` build:

- `What is entropy?`
- `Compare black hole and neutron star`
- `Tell me about the Great Attractor`
- `Run ANH` on a known FITS file
- `Run ANH` on a known MAST zip
- `Generate me 5 content ideas on topic black holes`
- `Design experiment to test whether light affects plant growth`
- `Generate concept for a lightweight propulsion system under these constraints: low mass, easy maintenance`
- `Model energy with mass 2 velocity 3 height 5`
- `Analyze orbit profile with semi-major axis 3 and eccentricity 0.2`
- attach a file/folder/zip from the desktop UI and confirm the path is handed off cleanly

The expected posture is:

- known topics answer directly when ambiguity is not critical
- ambiguous prompts clarify with readable options
- unknown topics refuse safely without hallucinating
- tool failures report what is missing rather than failing silently
