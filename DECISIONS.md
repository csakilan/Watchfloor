# Decisions

One line per decision: what changed, and why.

## 2026-09-10, from the design document

- Supervisor emits a template choice plus slot values instead of a free-text question. Free generation is unreliable at Flash tier.
- Synthesizer emits an ordinal confidence level; the numeric score is derived deterministically. Small models are badly calibrated and confident-wrong is the headline metric.
- Graph starts at a deterministic `preflight` node before `triage`. Enumerable work belongs in code.
- Added a Phase 0 capability spike. The weak-model adaptation is an untested bet and testing it in week one is cheap.
- Evaluation adds label pre-registration, a five-scenario held-out set, and external anchoring. Publishing the suite alone does not address overfitting during tuning.
- The single-agent baseline is also built on LangGraph, so the comparison differs in graph shape rather than in codebase.
- Default request rate set to 12 per minute, below the observed free-tier ceiling, to leave headroom for retries.
