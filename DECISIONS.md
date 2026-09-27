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

## 2026-09-27, Phase 0 plan amendments

A pre-flight scan before resuming Tasks 3 to 6 found defects in the plan's own text. Every amended code block was executed in a scratch copy of the repo before any implementer saw it: 60 unit tests and both integration tests passed, the integration run against a real kind cluster, with `~/.kube/config` verified byte-identical before and after.

- Ruling on the open Task 3 finding: `list_clusters` raises with kind's stderr when `kind get clusters` fails, and the missing tests for delete and for both `RuntimeError` paths are added. A failing listing used to read as "no clusters", so `delete_cluster` reported a clean no-op without having looked.
- Every `kind` call and every scenario script is pinned to a private kubeconfig at `.watchfloor/kubeconfig`. Unpinned, `kind create cluster` rewrites `~/.kube/config` and switches the user's context, and scenario scripts act on whichever cluster kubectl points at. This machine has `kind-miniargo` and `kind-nimbus` contexts, so the risk was live. An integration test now hashes `~/.kube/config` before and after.
- Scenario scripts run in their own process group, and the whole group is killed on timeout. Under a plain `subprocess.run` timeout only the script dies, and anything it started, a hung `kubectl` for instance, keeps running into the next scenario.
- Scenario 001's `break.sh` deletes the checkout pod instead of using `rollout restart`. With one replica, a rolling update keeps the old pod Ready until the new one is Ready, which never happens here, so the Service kept an endpoint and the pre-registered label's "no ready endpoints" was false. `verify.sh` now checks both halves of the label. `label.yaml` is unchanged: the fixture was corrected to match the pre-registered label, never the reverse.
- Scenario 001's checkout container now serves on 8080 via busybox `httpd`, so the healthy baseline actually answers on the port its Service targets. Closes the Task 2 deferred minor about port 8080.
- The model client sends the schema through `response_json_schema`. `google-genai` 2.22.0 documents `response_schema` as an OpenAPI 3.0 subset and `response_json_schema` as the field that accepts JSON Schema, which is what `model_json_schema()` produces. The plan had targeted the wrong field.
- The model client retries 429 and 5xx with exponential backoff, a 4 s base over 5 attempts, about a minute in total, and counts and rate-limits every attempt. The design required backoff on 429s and the plan had omitted it. The SDK's own retry is off by default, confirmed in `_api_client.retry_args`, so nothing is retried twice or retried where the counter cannot see it.
- `SlotFillDecision.specialist` is a `Literal` of the four specialists, per the constraint that every control-flow output is schema-constrained.
- `QuestionTemplate` refuses to load unless its declared slots and its placeholders match in both directions. The plan's test checked only that declared slots appeared in the text.
- The spike's `usable_rate` requires every identifier slot (namespace, pod, container, name, resource_name) to appear as a whole token in what the model was shown. A question about a pod the model invented is not usable, and before this change it would have counted as usable. `ungrounded_identifier_rate` is reported alongside it so nothing is hidden.
- The spike's prompt shows each template's question text, not only its id, so it measures the task the real supervisor will face.
- The spike is injectable and tested offline, and a provider that gives up mid-run stops it cleanly with partial results kept. The Phase 0 decision may not be made from an incomplete run.
- Test fixtures use placeholder model ids. Task 5's plan text repeated the real-model-id violation Task 1 was flagged for, and the global constraints now state that test code is included.
- Machine: 16 GB RAM, 8 cores, with Docker Desktop allotted about 7.7 GB and 4 CPUs. A 4-bit 8B model, about 5 GB, runs natively outside Docker's allotment, so local specialists stay viable. The decision waits on the spike.

## Open

- 2026-09-12, Task 2, deferred minor: `tests/test_labels.py` opens with a stray blank line, the same class of issue the Task 1 fix removed from `tests/test_model_routing.py`. For the final review to triage.
