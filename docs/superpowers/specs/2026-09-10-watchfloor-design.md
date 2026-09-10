# Watchfloor: Design and Build Plan

Date: 2026-09-10
Status: design agreed, ready for implementation planning
Supersedes: `PROJECT_SPEC.md` where the two conflict. Conflicts are listed in section 12.

---

## 1. What this document is

`PROJECT_SPEC.md` is the original build brief. It is a good document and most of it survives unchanged. This document exists because four constraints emerged after it was written, and each one moves something structural: a model-capability constraint, a mandatory framework, a credibility problem in the evaluation, and a decision about scope.

Every section below states the choice and the reason for it. Where a choice contradicts the original spec, that is called out explicitly rather than left for a reader to discover.

---

## 2. What Watchfloor is, in one paragraph

A system that diagnoses failures in a Kubernetes cluster. It receives a one-sentence symptom of the kind a human would actually report, and it determines the root cause by gathering evidence through read-only tools. A supervisor routes work to four specialists, each owning a narrow slice of the cluster's observable surface. The deliverable is not the agent. It is a reproducible evaluation showing how the multi-agent system performs against a single-agent baseline on a fixed suite of seeded failures.

---

## 3. The constraints that shaped this design

**No frontier model is available.** Inference runs on Gemini Flash-tier free allowances, possibly with a local Ollama model for volume. This is the constraint with the largest design consequence, and section 5.4 is entirely about absorbing it.

**LangGraph is mandatory.** This is not a problem, because the control flow has genuine structure: five-way conditional routing, two bounded loops, and state that accumulates across branches. LangGraph earns its place here in a way it would not in a linear pipeline.

**An agentic platform is mandatory.** Used as the tracing and deployment layer. The specific product is still open, see section 13.

**Scope is the full Kubernetes system.** The alternative was a smaller version over application logs, reaching a working agent in about a week. That was considered and rejected in favour of the full system, so the phase plan front-loads cluster tooling as the original spec intends.

---

## 4. Non-goals

Unchanged from the original spec, and each stays out for a stated reason.

| Excluded | Reason |
|---|---|
| Write operations of any kind | The system diagnoses, a human acts. Permanent constraint, not a v1 shortcut |
| Vector database or RAG | There is no retrieval problem here. Adding one would be unmotivated |
| Web UI | A CLI, traces, and a metrics table are the interface |
| Multi-cluster | One kind cluster |
| Agent-to-agent communication | Star topology only, see 5.3 |
| Fine-tuning | |
| Auto-remediation, even opt-in | |

---

## 5. Architecture

### 5.1 The shape

A single shared state object flows through a graph. Reducers handle concurrent appends.

```
START -> preflight -> triage -> supervisor

supervisor --conditional--> logs_specialist
                         |> events_specialist
                         |> config_specialist
                         |> state_specialist
                         |> synthesizer

<each specialist> -> supervisor

synthesizer --conditional--> supervisor   (low confidence and attempts < 2)
                          |> END
```

Two loops, both bounded. The evidence loop caps at eight dispatches, after which routing goes straight to the synthesizer and the verdict is marked capped. The confidence loop caps at two kickbacks, each of which appends a pseudo-evidence entry describing what was missing, so the supervisor has something concrete to act on rather than a vague instruction to try harder.

Both caps are non-negotiable. An agent that can spin is a bug.

### 5.2 The preflight node, which is new

Before any model call, run the sweep a human would type first: pods not in `Running` phase, `Warning` events in the last ten minutes, workloads reporting unready conditions. The result becomes evidence entry zero.

**Why.** This is enumerable work with no judgment in it, and enumerable work belongs in code. In most scenarios it identifies the namespace and the failing workload, which is the single hardest part of the supervisor's first decision and the part a weak model is least equipped to make from a one-sentence symptom. It costs nothing, it runs in milliseconds, and it is honest as long as the README says it exists.

The general principle, which recurs throughout this design: **put the model at the boundary between the messy world and the formal system, and keep it out of the formal system's interior.**

### 5.3 Why star topology

Specialists never talk to each other. All coordination goes through the supervisor.

Debuggability is the first reason. One transcript to read rather than four interleaved ones, which matters enormously when the thing you are debugging is why a routing decision was made.

Arbitration is the second. When two specialists produce conflicting evidence, exactly one node is responsible for resolving it. Peer topologies have no such node and the conflict surfaces as an inconsistent final answer with no obvious place to look.

Context integrity is the third. Handoff between peers is the most common place multi-agent systems silently lose information.

Adding a peer edge later is a small change and can be presented as an extension with a measured before and after.

### 5.4 Absorbing the weak-model constraint

This is where the design departs from the original spec most sharply. Everything else in that document survives a Flash-tier model. The supervisor does not, because its job is correlating evidence across surfaces and deciding what to gather next, which is deduction under sparse evidence, the one thing small models are genuinely poor at.

Four changes.

**Slot filling replaces free-form questions.** The original spec has the supervisor emit a free `question` string and prompts for specificity. A Flash-tier model writes "check the logs"; a local 8B model writes worse. Instead, each specialist owns a small set of question templates, and the supervisor chooses a template and fills its slots. Config templates look like `does {kind}/{name} in {namespace} reference a {resource_kind} that is missing or lacks key {key}`. Deterministic code composes the final question from the filled template.

Slot filling is reliable at this model tier in a way that open generation is not, and it has a second benefit: the template set is a readable enumeration of what the system is capable of asking, which makes failures interpretable.

**Discriminating power replaces open reasoning.** Rather than asking the supervisor to reason globally about the next step, ask it to score each of the four specialists on how well that specialist would distinguish between its current top two hypotheses, then take the maximum. Four small local judgments outperform one large global judgment at this tier, and the individual scores are inspectable when a routing decision looks wrong.

**Ordinal confidence replaces a float.** The original spec has the synthesizer emit `confidence: float`. Weak models are badly calibrated and will emit 0.9 for nearly everything, which destroys the confident-wrong metric that the entire project is built around. Two changes: the model picks from three ordinal levels rather than a continuous value, and a deterministic component is folded in. That component draws on how many hypotheses remain viable, what fraction of gathered evidence was marked relevant, and whether the specialists named in the scenario's `minimum_evidence` were actually consulted. The reported score is a function of both parts.

**Constrained decoding everywhere, with a deterministic fallback.** Every model output that drives control flow is an enum or a schema-constrained object, never parsed free text. If a decision fails to parse, or names a specialist that is not a valid edge key, fall through to a deterministic default rather than retrying, and assert loudly so it appears in the logs instead of silently degrading. The original spec warns that conditional edges can silently never fire when a routing function returns an unmatched string; this is the guard against that.

### 5.5 State

```python
class Evidence(BaseModel):
    specialist: Literal["preflight", "logs", "events", "config", "state"]
    tool_called: str
    tool_args: dict
    finding: str
    raw_excerpt: str | None
    relevant: bool

class Hypothesis(BaseModel):
    description: str
    supporting_evidence: list[int]
    contradicting_evidence: list[int]

class Verdict(BaseModel):
    root_cause_category: str        # from the fixed taxonomy
    explanation: str
    confidence_level: Literal["low", "medium", "high"]
    confidence_score: float         # derived, not model-emitted
    suggested_remediation: str
    evidence_cited: list[int]
    capped: bool
```

A specialist that finds nothing must record it with `relevant: false` rather than inventing a finding. This matters for the evidence-precision metric and it matters more for behaviour, since a model that believes it must always produce a finding will produce one.

---

## 6. The evidence layer

A custom MCP server in this repo, running as a local stdio server, exposing roughly fourteen read-only tools across the four surfaces: logs, events, config, and live state.

Three properties are load-bearing.

**Read-only, permanently.** No patch, no delete, no exec, under any framing. The tool for reading Secrets returns key names and value lengths only, never values, and a test asserts this.

**Flat argument schemas.** Primitives and simple string enums only. No nested objects, no `anyOf`. Gemini's function-calling support is stricter than some providers about nested JSON Schema, and flat schemas sidestep the problem while being better design regardless.

**Recoverable error messages.** A tool that fails must return something the agent can act on. Not `Error: 404`, but:

```
Pod 'checkout-7d9f' not found in namespace 'shop'.
Pods present in this namespace: checkout-8a2b, redis-0, worker-4c1d.
```

The failure carries its own recovery, so the agent does not have to guess and burn an iteration. This measurably improves agent performance and it deserves a paragraph in the README, because it is the least obvious lever in the whole system.

---

## 7. Evaluation

This is the deliverable. Everything above is machinery in service of it.

### 7.1 Scenarios

Each scenario is a directory containing a healthy baseline manifest, a `break.sh`, a `verify.sh` that exits zero when the fault is live as intended, a `cleanup.sh`, and a `label.yaml` carrying the ground truth.

Fifteen scenarios spread across a fixed taxonomy of fifteen root-cause categories, so grading is deterministic string matching rather than judgment. Difficulty tiers: five easy where one specialist suffices, seven medium requiring correlation across two, three hard where the obvious signal is a red herring and the real cause is elsewhere.

At least eight scenarios carry a deliberate distractor. A suite without distractors measures nothing, because a single tool call would solve every case.

### 7.2 Credibility, which needs more than the original spec gives it

The original spec acknowledges honestly that the scenarios are self-authored and therefore self-graded, and proposes publishing the suite as the mitigation. That is necessary and not sufficient. Three additions.

**Pre-registration through git.** Author and commit all fifteen `label.yaml` files, and the taxonomy, before running any agent against them. Git history then proves the labels were not adjusted to match what the system produced. This costs nothing and converts an assertion of good faith into a verifiable fact.

**A held-out set.** Freeze five of the fifteen scenarios and do not look at their results during prompt tuning. Tune against the other ten. Report both numbers separately in the README. Prompt tuning against your whole suite is overfitting, and it is the most common unexamined flaw in portfolio evaluation work. Reporting a held-out number is rare enough to be a differentiator on its own.

**External anchoring.** Derive as many scenarios as possible from documented real-world Kubernetes failure modes rather than inventing all fifteen, and cite the source in `label.yaml`. This moves the suite from "faults I imagined" toward "faults that are known to happen."

**Publish the failures.** The README must state which scenarios the system fails and what the pattern in those failures is. A README claiming fifteen out of fifteen is less credible than one that says the supervisor commits to its first hypothesis and does not seek disconfirming evidence.

### 7.3 Metrics

Three primary, computed by a scoring script from committed results JSON.

**Root cause accuracy.** Predicted category equals the label. Reported overall, by difficulty tier, and split between the tuned set and the held-out set.

**Dispatches to resolution.** Specialist invocations before the verdict. Lower is better at equal accuracy. Mean and distribution.

**Confident-wrong rate.** Fraction of runs where confidence is `high` and the category is wrong. This is the headline. In an operations context a confidently wrong diagnosis is worse than an admitted unknown, because someone acts on it.

Secondary: tokens and estimated cost per run, wall clock per run, cap-hit rate, and evidence precision measured as the fraction of gathered evidence marked relevant.

### 7.4 The baseline

Every metric is reported against a single-agent baseline: one agent, all tools, no supervisor, same iteration cap, same scenarios, same model. Only one variable changes, which is whether the work is split up.

The baseline is built in Phase 2, before the multi-agent system. Three reasons. Order protects honesty, because a baseline built afterwards becomes a chore you half-build to make your real work look good. It forces the measuring apparatus into existence, since the runner, the scorer, and the results format all have to work before any agent complexity arrives. And it can falsify the premise cheaply: if one agent solves fourteen of fifteen, the problem does not need a supervisor and you have learned that in week two.

The baseline is also built on LangGraph, as a single node with a tool loop. This keeps the runner interface uniform and makes the comparison a difference in graph shape rather than a difference in codebase.

### 7.5 Trials

Three trials per scenario for final measurement, one during iteration. Report mean and standard deviation. Agents are non-deterministic and a single-trial number is not a result.

---

## 8. Stack

| Layer | Choice | Note |
|---|---|---|
| Language | Python 3.12 | |
| Orchestration | LangGraph, checkpointer from day one | Mandatory, and justified by the control flow |
| Tool protocol | Custom MCP server, Python `mcp` SDK | Written here, not consumed |
| MCP to LangChain bridge | `langchain-mcp-adapters` | |
| Cluster | kind and kubectl | |
| Cluster access | Official `kubernetes` client, inside the MCP server only | The agent never touches it directly |
| Inference | Gemini Flash tier, free allowance | Verify the current SDK name before installing |
| Local inference | Ollama, 8B class, for specialists | Conditional on available RAM, see section 13 |
| Structured output | Pydantic plus provider-native schema enforcement | Never parse free text for a control decision |
| Tracing and platform | Open, see section 13 | |
| Eval | pytest plus a custom scoring script | Grading is deterministic, so no eval framework |
| Config | Pydantic Settings plus `config/models.yaml` | |

Model selection is never hardcoded in node logic. It lives in `config/models.yaml` keyed by node role, which makes the cross-provider comparison in Phase 5 a config change rather than a refactor.

---

## 9. Cost and rate budget

Worth computing before the build rather than discovering in week four.

A baseline run is roughly eight model calls. A multi-agent run is roughly thirty, because every specialist tool use is another round trip and the supervisor is consulted between each dispatch.

A full final measurement is fifteen scenarios times three trials times two agents, which is ninety runs and approximately **1,700 requests**.

Against a free tier of roughly fifteen requests per minute and on the order of 1,500 requests per day, that is slightly more than one full day of quota for a single complete pass, and about two hours of wall clock at the rate limit even with no other delay. Survivable for a final pass. Not survivable for iteration, which is why the harness must support running subsets and why a request counter is logged per run.

If paid, the same pass costs on the order of five dollars, and the entire six-week project lands somewhere around thirty to sixty dollars. Treat those as order-of-magnitude figures. The point is that the free tier costs time rather than money, and the decision of which to spend should be made deliberately.

Mitigations built in from the start: a shared rate limiter with exponential backoff on 429s, a per-run request counter so a suite pass can be estimated before it is launched, and subset execution in the harness.

---

## 10. Repo layout

```
Watchfloor/
  README.md
  DECISIONS.md
  pyproject.toml
  docs/superpowers/specs/     # this document
  config/
    models.yaml
    settings.py
  mcp_server/
    server.py
    k8s_client.py
    errors.py
    tools/{logs,events,config,state}.py
  agent/
    graph.py
    state.py
    nodes/{preflight,triage,supervisor,specialists,synthesizer}.py
    prompts/*.md
    templates/questions.yaml    # the slot-fill templates
    models.py
  baseline/single_agent.py
  scenarios/001-.../ ... 015-.../
  eval/{runner,score,report}.py
  results/*.json
  tests/
```

Prompts live in `.md` files rather than string literals, so prompt changes are visible in `git diff`. This matters because Phase 4 requires logging every prompt version against its metric delta, and a diff is the cheapest way to make that record trustworthy.

Question templates live in YAML for the same reason.

---

## 11. Phases

Three principles govern the breakdown.

**Build the thing that can say "wrong" before the thing that can be wrong.** Evaluation infrastructure precedes agent sophistication throughout.

**Every phase ends in something runnable and measured.** No phase ends with "the code exists."

**Touch the largest unknown earliest.** A risk discovered in week one is a design change. The same risk discovered in week four is a rewrite.

### Phase 0: foundations and the model spike (2 to 3 days)

Repo skeleton, `pyproject.toml`, `config/models.yaml`, settings. An idempotent kind cluster creation script. One complete scenario with all four scripts and its label.

And the piece that is not in the original spec: a **model capability spike**. Fifty trials asking the target model to choose from an enum and fill a slot template, measuring the parse failure rate and the rate of nonsensical slot values.

*Why this phase exists.* The weak-model constraint is the single largest threat to this design, and section 5.4 is an untested bet against it. Fifty API calls and half a day tell you whether that bet holds. If Flash-Lite cannot fill a template reliably, the model routing plan changes, and changing it in week one is free while changing it in week three is not.

**Acceptance.** Cluster comes up and tears down repeatably. One scenario breaks and its `verify.sh` exits zero. The spike produces a number, and that number is recorded in `DECISIONS.md`.

### Phase 1: the evidence layer (week 1)

The MCP server with at least eight tools working, read-only, flat schemas, recoverable error messages, and the Secret redaction test. Five scenarios spanning the difficulty tiers, weighted toward easy. A manual script that runs every tool against a broken cluster and prints the output.

*Why this phase exists here.* Tools before agents, because a tool that returns unusable output produces an agent that looks stupid, and you will spend days debugging the agent before suspecting the tool. The gate below prevents that misattribution entirely.

**Acceptance.** You break the cluster, call every tool by hand, and can personally diagnose the fault from the output alone. If a human cannot, no agent will.

### Phase 2: baseline and harness (week 2)

One agent, all tools, iteration cap, structured verdict with ordinal confidence. `eval/runner.py` looping scenarios and writing JSON. `eval/score.py` computing all three primary metrics. The rate limiter and the request counter. First measured numbers on five scenarios.

*Why before the multi-agent system.* Covered in 7.4. In short: order protects honesty, the harness has to exist anyway, and the premise might be falsified cheaply.

**Acceptance.** `python -m eval.runner --agent baseline --scenarios all` produces a metrics table. The numbers do not need to be good. The request count per run is known and logged.

### Phase 3: the multi-agent graph (week 3)

The hard week. Preflight node. Supervisor with enum-constrained decisions and slot-filled questions. Four specialists with scoped tool access. Both loop-backs with working caps. Checkpointing. The edge assertion from 5.4.

*Why third.* It depends on tools that work, a harness that measures, and a baseline to beat, all of which now exist. Building it first would mean debugging three things at once with no way to tell which was broken.

**Acceptance.** A run visibly dispatches to at least two different specialists on a medium scenario, and the confidence loop fires at least once across the suite. **Verified by reading the checkpoint history, not by trusting the final answer.** A correct verdict can come from a graph that never dispatched anything, and only the state history distinguishes the two.

### Phase 4: observability, full suite, tuning (week 4)

The agentic platform wired in, with spans for every node and every tool call. Expansion to fifteen scenarios with red herrings, pre-registered per 7.2. Full suite, three trials, both agents. Prompt tuning against the ten tuning scenarios only, with every prompt version logged in `DECISIONS.md` alongside its metric delta.

*Why tracing lands here rather than earlier.* Before Phase 3 there is little to trace and reading a JSONL log by eye is faster. After Phase 3 the interleaving of supervisor decisions and specialist calls becomes genuinely hard to follow by hand, which is the point at which a tracing product starts paying for itself.

**Acceptance.** A committed results file covering both agents, fifteen scenarios, three trials, with the metrics table broken out by difficulty tier and by tuned versus held-out.

### Phase 5: comparison and polish (weeks 5 to 6)

Swap `config/models.yaml` to run the same graph on a second provider or a local Ollama model, and report the comparison table. Write the README in the order specified in section 10 of the original spec: one sentence on what it is, then the metrics table, then the architecture diagram, then the failure analysis, then design decisions and exclusions, then honest limitations, and setup instructions last. Record a two to three minute terminal demo.

*Why the model swap is a phase rather than an afterthought.* It is the payoff for keeping model selection in config from day one, and it produces a second measured comparison at almost no cost. Two comparisons in one project reads very differently from one.

**Acceptance.** A stranger reads the README and understands what was built, how it was measured, and what did not work.

---

## 12. Where this contradicts `PROJECT_SPEC.md`

| Original | Now | Reason |
|---|---|---|
| Supervisor emits a free-text `question` | Chooses and fills a slot template | Free generation is unreliable at this model tier |
| `Verdict.confidence: float` from the model | Ordinal level from the model, score derived deterministically | Small models are badly calibrated, and this metric is the headline |
| Graph starts at `triage` | Graph starts at `preflight` | Enumerable work belongs in code, not in a model call |
| Five phases | Six, with a Phase 0 spike | The weak-model bet needs testing in week one |
| Suite is published as the credibility mitigation | Plus pre-registration, a held-out set, and external anchoring | Publishing alone does not address overfitting during tuning |
| Baseline is plain, framework unspecified | Baseline also on LangGraph | Keeps the comparison to one variable |

Each of these gets a one-line entry in `DECISIONS.md` when implemented.

---

## 13. Open decisions

**The agentic platform.** Requirement is known, product is not. Langfuse self-hosted via docker compose is what the original spec names and it is free. LangSmith pairs natively with LangGraph and needs less wiring. Resolve before Phase 4.

**Local model for specialists.** Depends on available RAM, which is currently unknown. Specialists are the overwhelming majority of request volume, so moving them local is the single largest lever on the rate budget. Resolve during Phase 0, informed by the spike.

**Confidence threshold for the synthesizer kickback.** The original spec proposes 0.6. Tune empirically in Phase 4 against the tuning set only.

**A fifth specialist for network checks, or fold into `state`.** Defer to Phase 3, decide from what the scenarios actually need.

**The name.** `Watchfloor` currently collides with twelve public GitHub repositories, several of them active and some in adjacent domains. Not blocking, but the original spec asked the question, so it deserves a recorded answer.

---

## 14. Decision log practice

`DECISIONS.md` is part of the deliverable, not overhead. Every entry is one line: what changed, and why. It captures the six contradictions above, every prompt version with its metric delta, the Phase 0 spike result, and each open decision as it resolves.

The reason this matters beyond tidiness: the README's credibility rests on the claim that the evaluation was not tuned to flatter the system, and a dated, committed record of what changed and when is the evidence for that claim.
