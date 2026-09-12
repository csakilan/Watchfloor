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

## Open, awaiting a ruling

- 2026-09-12, Task 3: `cluster/manage.py` `list_clusters` parses `kind get clusters` stdout without checking the exit code, so a failing `kind` call reads as "no clusters exist". `delete_cluster` then returns False and reports a clean no-op without having looked, while `create_cluster` and `delete_cluster` both raise with stderr attached two lines below. Raised by the Task 3 review as Important and plan-mandated, since the code came verbatim from the Phase 0 plan rather than from the implementer. Candidate fix: raise `RuntimeError` with kind's stderr on a non-zero listing exit, matching the sibling calls.
- 2026-09-12, Task 3, deferred minor: no test covers `delete_cluster` actually running `kind delete cluster`, nor either `RuntimeError` path. Same origin, the test list in the plan.
- 2026-09-12, Task 2, deferred minor: `tests/test_labels.py` opens with a stray blank line, the same class of issue the Task 1 fix removed from `tests/test_model_routing.py`.
- 2026-09-12, Task 2, deferred minor: scenario 001's Service targets port 8080 but the busybox container never listens, so the "healthy" baseline never serves the traffic its incident text describes. Harmless while verification reads pod status, relevant if a later task checks the scenario with a real request.
