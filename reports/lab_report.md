# Day 08 Lab Report

## 1. Student

- Name: Nguyen Duc Trong
- Project: LangGraph Agentic Orchestration

## 2. Architecture

The workflow starts with intake and LLM structured classification. Conditional edges route
simple requests to an LLM answer, read-only lookups through tool evaluation, incomplete requests
to clarification, risky mutations through approval, and system failures into a bounded retry loop.
Every terminal branch passes through `finalize` before `END`.

## 3. State schema

`route`, `risk_level`, `attempt`, `evaluation_result`, `pending_question`, `proposed_action`,
`approval`, and `final_answer` use overwrite semantics. `messages`, `tool_results`, `errors`, and
`events` use the additive reducer and form append-only audit trails. State values remain lean and
JSON-serializable so memory and SQLite checkpointers can persist them.

## 4. Metrics summary

| Metric | Value |
|---|---:|
| Total scenarios | 7 |
| Success rate | 100.0% |
| Average nodes visited | 6.43 |
| Total retries | 3 |
| Approval/HITL visits | 2 |
| Resume success | Not measured |

## 5. Scenario results

| Scenario | Expected | Actual | Success | Retries | Approval observed |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | Yes | 0 | No |
| S02_tool | tool | tool | Yes | 0 | No |
| S03_missing | missing_info | missing_info | Yes | 0 | No |
| S04_risky | risky | risky | Yes | 0 | Yes |
| S05_error | error | error | Yes | 2 | No |
| S06_delete | risky | risky | Yes | 0 | Yes |
| S07_dead_letter | error | error | Yes | 1 | No |

## 6. Failure analysis

1. **Transient tool failure:** results containing `ERROR` are rejected by evaluation. The retry
   node increments a persisted counter, and routing sends the request to dead letter once
   `attempt >= max_attempts`, preventing infinite loops.
2. **Unsafe side effect:** refund, deletion, cancellation, email, and similar requests are prepared
   but never executed before the approval node. Rejection routes to clarification instead.
3. **Model or provider failure:** an absent/invalid API credential fails explicitly rather than
   silently substituting scenario-specific answers. Production should add provider-level timeout,
   backoff, and a monitored fallback model.

## 7. Persistence and recovery

Every invocation supplies a stable `thread_id`. MemorySaver supports tests, while the SQLite
adapter enables durable checkpoints with WAL mode. The compiled graph accepts either checkpointer,
allowing state history inspection and resumption with the same thread configuration.

## 8. Extension work

Implemented SQLite persistence and optional real human-in-the-loop execution. Setting
`LANGGRAPH_INTERRUPT=true` pauses at `interrupt()` and accepts a human approval payload; the
default deterministic mock approval keeps automated grading non-interactive.

## 9. Improvement plan

Production work should replace the mock tool with authenticated, idempotent APIs; add policy-based
authorization and approval expiry; use an LLM judge with deterministic safeguards; capture real
latency/token metrics; add tracing and adversarial classification tests; and encrypt checkpoint
data with retention controls.
