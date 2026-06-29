# Day 08 Lab Report

## 1. Team / student

- Name: Mai Đức Vinh
- Repo/commit: 6d8252d
- Date: 2026-06-29

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
- **Total Scenarios**: 7
- **Success Rate**: 100.00%
- **Average Nodes Visited**: 6.43
- **Total Retries**: 3
- **Total Interrupts**: 2

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | ✅ Yes | 0 | 0 |
| S02_tool | tool | tool | ✅ Yes | 0 | 0 |
| S03_missing | missing_info | missing_info | ✅ Yes | 0 | 0 |
| S04_risky | risky | risky | ✅ Yes | 0 | 1 |
| S05_error | error | error | ✅ Yes | 2 | 0 |
| S06_delete | risky | risky | ✅ Yes | 0 | 1 |
| S07_dead_letter | error | error | ✅ Yes | 1 | 0 |

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
