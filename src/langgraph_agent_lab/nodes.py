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

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event


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


class Classification(BaseModel):
    route: str = Field(description="The classification route. Must be one of: simple, tool, missing_info, risky, error")
    risk_level: str = Field(description="The risk level. Must be 'high' for risky, and 'low' for others")


class Evaluation(BaseModel):
    needs_retry: bool = Field(description="True if the tool result indicates a failure or error requiring retry. False if successful.")


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.
    """
    query = state.get("query", "").strip()

    # Keyword fallback for resilience
    lower_query = query.lower()
    fallback_route = "simple"
    fallback_risk = "low"
    if any(k in lower_query for k in ["refund", "delete", "cancel"]):
        fallback_route = "risky"
        fallback_risk = "high"
    elif any(k in lower_query for k in ["lookup", "order", "status", "track"]):
        fallback_route = "tool"
    elif "fix" in lower_query and len(lower_query.split()) < 5:
        fallback_route = "missing_info"
    elif any(k in lower_query for k in ["timeout", "failure", "crash", "error"]):
        fallback_route = "error"

    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(Classification)
        prompt = (
            "You are a customer support ticket routing agent. Classify the user query into exactly one of the following routes:\n"
            "- risky: Actions with side effects like refunds, deletions, sending emails, cancellations.\n"
            "- tool: Information lookups like order status, tracking, database searches.\n"
            "- missing_info: Vague or incomplete queries lacking actionable context (e.g. 'can you fix it?', 'it does not work').\n"
            "- error: System failures, timeouts, crashes, service unavailable.\n"
            "- simple: General questions answerable without tools or actions (e.g. 'how to reset password', 'what is your email').\n\n"
            "Priority Guide: risky > tool > missing_info > error > simple. If a query could belong to multiple, choose the highest priority one.\n\n"
            f"Query: {query}"
        )
        res = structured_llm.invoke(prompt)
        route = res.route.strip().lower()
        risk_level = res.risk_level.strip().lower()
        if route not in ["simple", "tool", "missing_info", "risky", "error"]:
            route = fallback_route
        if risk_level not in ["high", "low"]:
            risk_level = "high" if route == "risky" else "low"
    except Exception:
        route = fallback_route
        risk_level = fallback_risk

    return {
        "route": route,
        "risk_level": risk_level,
        "events": [make_event("classify", "completed", f"Route classified as {route} with risk {risk_level}")],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.
    """
    attempt = state.get("attempt", 0)
    route = state.get("route", "")
    query = state.get("query", "")

    # If route is "error" and attempt < 2: simulate transient failure
    if route == "error" and attempt < 2:
        result = f"ERROR: Tool execution failed due to system error (attempt {attempt})"
    else:
        result = f"Success: Mock tool retrieved data for query '{query}' (attempt {attempt})"

    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", f"Tool output: {result}")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.
    """
    tool_results = state.get("tool_results", [])
    latest_result = tool_results[-1] if tool_results else ""

    # Heuristic default
    if "ERROR" in latest_result:
        eval_res = "needs_retry"
    else:
        eval_res = "success"

    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(Evaluation)
        prompt = (
            "Evaluate the following tool execution result. Decide if the execution failed/erred and needs to be retried.\n\n"
            f"Tool Result: {latest_result}\n\n"
            "Return true for needs_retry if there is a clear error/failure, or false if it succeeded."
        )
        res = structured_llm.invoke(prompt)
        eval_res = "needs_retry" if res.needs_retry else "success"
    except Exception:
        pass

    return {
        "evaluation_result": eval_res,
        "events": [make_event("evaluate", "completed", f"Evaluation result: {eval_res}")],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***
    """
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")

    context = []
    if tool_results:
        context.append(f"Tool results: {tool_results}")
    if approval:
        context.append(f"Approval details: {approval}")
    context_str = "\n".join(context)

    try:
        llm = get_llm(temperature=0.2)
        prompt = (
            "You are a helpful customer support agent. Answer the user's query based on the provided context (such as tool execution results or approvals, if any).\n"
            "Be professional, direct, and helpful. Do not mention system details like nodes, routing, or the graph.\n\n"
            f"User Query: {query}\n"
            f"Context:\n{context_str}\n\n"
            "Response:"
        )
        res = llm.invoke(prompt)
        answer = res.content.strip()
    except Exception:
        answer = f"I have processed your request '{query}' successfully. Context: {context_str}"

    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "Generated response grounded in context")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.
    """
    query = state.get("query", "")

    try:
        llm = get_llm(temperature=0.2)
        prompt = (
            "The user query is vague or incomplete. Generate a polite and helpful clarification question asking the user to provide the missing details.\n\n"
            f"Vague query: {query}\n\n"
            "Clarification Question:"
        )
        res = llm.invoke(prompt)
        question = res.content.strip()
    except Exception:
        question = f"Could you please provide more details or clarify what you mean by '{query}'?"

    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", f"Clarification requested: {question}")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.
    """
    query = state.get("query", "")
    proposed_action = f"Perform high-risk operation: '{query}'"
    return {
        "proposed_action": proposed_action,
        "events": [make_event("risky_action", "completed", f"Prepared action: {proposed_action}")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.
    """
    proposed_action = state.get("proposed_action", "")

    if os.getenv("LANGGRAPH_INTERRUPT") == "true":
        from langgraph.types import interrupt
        decision = interrupt(
            {
                "proposed_action": proposed_action,
                "prompt": "Approve this risky action?"
            }
        )
        if isinstance(decision, dict):
            approved = decision.get("approved", False)
            comment = decision.get("comment", "")
            reviewer = decision.get("reviewer", "human-reviewer")
        else:
            approved = bool(decision)
            comment = str(decision)
            reviewer = "human-reviewer"
    else:
        approved = True
        comment = "Mock approval by default (non-interactive mode)"
        reviewer = "mock-reviewer"

    approval_dict = {
        "approved": approved,
        "reviewer": reviewer,
        "comment": comment
    }
    event_msg = f"Action {'approved' if approved else 'rejected'} by {reviewer}: {comment}"

    return {
        "approval": approval_dict,
        "events": [make_event("approval", "completed", event_msg)],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.
    """
    attempt = state.get("attempt", 0)
    new_attempt = attempt + 1
    err_msg = f"Attempt {new_attempt} failed."
    return {
        "attempt": new_attempt,
        "errors": [err_msg],
        "events": [make_event("retry", "completed", f"Attempt counter incremented to {new_attempt}")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.
    """
    query = state.get("query", "")
    msg = f"We apologize, but we were unable to process your request '{query}' after multiple attempts."
    return {
        "final_answer": msg,
        "events": [make_event("dead_letter", "completed", "Request moved to dead letter queue")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
