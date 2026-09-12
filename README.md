# Watchfloor

Multi-agent Kubernetes incident diagnosis, measured against a single-agent baseline.

## The problem

- Symptoms surface far from causes. A missing ConfigMap key blocks a container, so the Deployment has no ready pods, so the Service has no endpoints, so callers get 503s.
- The 503 says nothing about ConfigMaps.
- Evidence lives on four surfaces that barely overlap: logs, cluster events, config, live state.
- Diagnosing means knowing which surface to check, and correlating two of them when one is ambiguous.

## How it works

Input is one sentence, such as `"the checkout service is returning 503s"`. No namespace, no workload, no hint at what changed.

- **Four specialists**, one per evidence surface. Each can only use its own tools.
- **One supervisor** decides which specialist to dispatch, what to ask it, and when to stop.
- **No specialist-to-specialist contact.** One transcript to read, and one node responsible when evidence conflicts.
- **Read-only, always.** It diagnoses, a human acts.
- **Hard caps.** 8 dispatches, 2 low-confidence retries.

```
incident -> supervisor -> specialist -> supervisor
                      -> synthesizer -> verdict
```

## How it is evaluated

- 15 scenarios break the cluster in a known way, so the right answer exists before the run.
- Scoring is exact category matching against a fixed taxonomy. No LLM judges.
- Most scenarios carry a deliberate distractor.
- Everything is compared against a single-agent baseline with identical tools, model and caps.

| Metric | Why it matters |
|---|---|
| Root cause accuracy | Overall and by difficulty tier |
| Dispatches to resolution | Lower is better at equal accuracy |
| Confident-wrong rate | Headline metric. A confident wrong answer is worse than an admitted unknown, because someone acts on it |
