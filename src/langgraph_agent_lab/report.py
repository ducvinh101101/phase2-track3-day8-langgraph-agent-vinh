"""Report generation helper.

TODO(student): implement report rendering using MetricsReport data
and the template in reports/lab_report_template.md.
"""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data.

    Generate a report that includes:
    1. Metrics summary table (total scenarios, success rate, retries, interrupts)
    2. Per-scenario results table
    3. Architecture explanation (your graph design, state schema, reducers)
    4. Failure analysis (at least two failure modes you considered)
    5. Improvement plan
    """
    import subprocess
    from datetime import datetime

    commit = "unknown"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        pass

    date_str = datetime.now().strftime("%Y-%m-%d")

    # Render scenarios table
    scenario_rows = []
    for s in metrics.scenario_metrics:
        success_str = "✅ Yes" if s.success else "❌ No"
        scenario_rows.append(
            f"| {s.scenario_id} | {s.expected_route} | {s.actual_route} | {success_str} | {s.retry_count} | {s.interrupt_count} |"
        )
    scenarios_table = "\n".join(scenario_rows)

    report_md = f"""# Day 08 Lab Report

## 1. Team / student

- Name: Mai Đức Vinh
- Repo/commit: {commit}
- Date: {date_str}

## 2. Architecture

Our graph implements a production-grade support-ticket agent using LangGraph. The design utilizes a directed acyclic architecture with cyclical retry loops and interactive human-in-the-loop (HITL) checkpoints.

Key Nodes:
1. `intake`: Normalizes and prepares the raw input query.
2. `classify`: Uses a Gemini LLM with structured output (`with_structured_output`) to classify the ticket's intent and priority.
3. `tool`: Executes operations, simulating transient system/timeout errors when necessary.
4. `evaluate`: Uses LLM-as-judge to verify if tool execution succeeded or requires retry.
5. `risky_action`: Formulates descriptions for high-risk actions.
6. `approval`: Triggers interactive human approval (or fallback mock) using LangGraph's `interrupt`.
7. `retry`: Tracks attempt counters and records transient errors.
8. `dead_letter`: Handles max attempt limit exhaustion gracefully.
9. `clarify`: Requests additional context for vague tickets.
10. `answer`: Synthesizes final grounded responses using LLM based on query & tool contexts.
11. `finalize`: Audits and marks the final workflow completion.

## 3. State schema

State schema is built on a TypedDict representing the agent context:

| Field | Reducer | Why |
|---|---|---|
| messages | append | audit conversation/events |
| route | overwrite | current route only |
| attempt | overwrite | current retry attempt count |
| max_attempts | overwrite | maximum retry attempt count permitted |
| final_answer | overwrite | final synthesized answer |
| pending_question | overwrite | clarification question |
| proposed_action | overwrite | risky action description |
| approval | overwrite | human-in-the-loop approval decision |
| tool_results | append | tool outputs history |
| errors | append | error messages history |
| events | append | audit log events |

## 4. Scenario results

Here is the metrics summary:
- **Total Scenarios**: {metrics.total_scenarios}
- **Success Rate**: {metrics.success_rate:.2%}
- **Average Nodes Visited**: {metrics.avg_nodes_visited:.2f}
- **Total Retries**: {metrics.total_retries}
- **Total Interrupts**: {metrics.total_interrupts}

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
{scenarios_table}

## 5. Failure analysis

We carefully handled and tested two primary failure modes:

1. **Retry or tool failure**:
   Transient errors during tool execution (e.g., S05/S07 scenarios simulating timeouts) are routed to a cyclic loop. The `retry_or_fallback_node` increments the attempt counter. If the counter is less than `max_attempts`, the graph retries the tool; otherwise, it escalates to the `dead_letter_node` preventing infinite loops.

2. **Risky action without approval**:
   Deletions, refunds, and other destructive actions are classified as `risky` and immediately routed to `risky_action` and `approval_node`. Under production settings, the checkpointer persists state and `interrupt()` pauses execution until a human reviewer sends an approval payload. If rejected, it routes to clarification instead of executing the tool.

## 6. Persistence / recovery evidence

We implemented `SqliteSaver` inside `persistence.py`. A local SQLite connection configured with Write-Ahead Logging (`WAL` mode) persists agent state checkpoints. Running scenarios with unique `thread_id` values ensures distinct state tracking and allows state history querying and resumption after interruptions.

## 7. Extension work

We completed the following bonus extensions:
1. **SQLite Persistence**: Fully wired `SqliteSaver` which stores execution checkpoint histories in `state_checkpoints.db`.
2. **Interactive HITL**: Configured real interrupts using `interrupt()` when `LANGGRAPH_INTERRUPT=true` is set.
3. **LLM-as-Judge**: Implemented structured quality-check evaluation of tool results in the `evaluate_node`.

## 8. Improvement plan

If we had one more day to productionize this workflow, we would:
1. Implement parallel fan-out/fan-in using the `Send` API to execute multiple independent tool lookups concurrently.
2. Build a full web dashboard/UI (using Streamlit or Next.js) to view current checkpointer logs, active interrupts, and allow admin approvals/rejections in real time.
3. Hook up LangSmith tracing to monitor model latencies, costs, and output quality across versions.
"""
    return report_md


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
