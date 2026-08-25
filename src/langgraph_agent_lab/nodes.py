"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event


class Classification(BaseModel):
    """Strict schema returned by the classifier model."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    reasoning: str = Field(description="Brief reason for the selected route")


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── TODO(student): implement ALL nodes below ────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    prompt = f"""You route customer-support requests. Return exactly one route.
Priority (first matching category wins):
1. risky: requests to change external state, including refunds, deletion, cancellation,
   sending messages/email, payments, or account changes.
2. tool: read-only lookup, tracking, search, or retrieving specific records.
3. missing_info: vague or incomplete requests without enough context to act.
4. error: reports of timeouts, crashes, unavailable services, or processing failures.
5. simple: general informational questions answerable without a tool.
Classify by intent and never rely on scenario identifiers.

Customer request: {query!r}"""
    decision = get_llm(temperature=0).with_structured_output(Classification).invoke(prompt)
    route = decision.route
    return {
        "route": route,
        "risk_level": "high" if route == "risky" else "low",
        "events": [make_event("classify", "completed", "request classified", route=route)],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0))
    route = state.get("route", "")
    if route == "error" and attempt < 2:
        result = f"ERROR: transient support service failure on attempt {attempt + 1}"
        event_type = "failed"
    elif route == "risky":
        action = state.get("proposed_action") or state.get("query", "requested action")
        result = f"SUCCESS: approved support action completed: {action}"
        event_type = "completed"
    else:
        result = f"SUCCESS: support lookup completed for: {state.get('query', '')}"
        event_type = "completed"
    return {
        "tool_results": [result],
        "events": [make_event("tool", event_type, "mock tool executed", attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    results = state.get("tool_results", [])
    latest = results[-1] if results else "ERROR: no tool result"
    evaluation = "needs_retry" if "ERROR" in latest.upper() else "success"
    return {
        "evaluation_result": evaluation,
        "events": [make_event("evaluate", "completed", "tool result evaluated", result=evaluation)],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    context = "\n".join(state.get("tool_results", [])) or "No tool was required."
    approval = state.get("approval")
    prompt = f"""Write a concise, helpful customer-support response.
Use only the supplied context for factual or action-completion claims. Do not invent IDs,
statuses, policies, or completed actions. If no tool was required, answer the general question
from ordinary support knowledge. Mention approval only when it is supplied.

Request: {state.get("query", "")}
Tool context: {context}
Approval: {approval if approval is not None else "not applicable"}"""
    response = get_llm(temperature=0).invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    if isinstance(answer, list):
        answer = " ".join(str(part) for part in answer)
    return {
        "final_answer": str(answer).strip(),
        "events": [make_event("answer", "completed", "grounded response generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    approval = state.get("approval")
    if approval is not None and not approval.get("approved", False):
        question = (
            "The proposed action was not approved. What safe alternative would you like us to take?"
        )
    else:
        question = (
            f"Could you provide the affected account, order, or feature and describe the expected "
            f"outcome for your request: {state.get('query', '')!r}?"
        )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    action = (
        f"Execute the requested state-changing support operation only after human review: "
        f"{state.get('query', '')}"
    )
    return {
        "proposed_action": action,
        "events": [make_event("risky_action", "prepared", "action awaiting approval")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Returns an approval mapping and an append-only audit event.
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "false").lower() == "true":
        from langgraph.types import interrupt

        supplied = interrupt(
            {
                "question": "Approve this support action?",
                "proposed_action": state.get("proposed_action"),
            }
        )
        if isinstance(supplied, bool):
            decision = ApprovalDecision(approved=supplied, reviewer="human")
        else:
            decision = ApprovalDecision.model_validate(supplied)
    else:
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="Automatically approved for repeatable lab execution.",
        )
    return {
        "approval": decision.model_dump(),
        "events": [
            make_event(
                "approval", "approved" if decision.approved else "rejected", decision.comment
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0)) + 1
    error = f"Transient failure recorded; retry counter is now {attempt}"
    return {
        "attempt": attempt,
        "errors": [error],
        "events": [make_event("retry", "scheduled", error, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    attempts = int(state.get("attempt", 0))
    answer = (
        "We could not complete this request after "
        f"{attempts} attempt(s). It has been placed in the support dead-letter queue for review."
    )
    return {
        "final_answer": answer,
        "events": [
            make_event("dead_letter", "escalated", "retry limit exhausted", attempts=attempts)
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
