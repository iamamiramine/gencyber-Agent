"""Does a GUARD-BLOCKED tool call still charge the action budget?

Before the fix it did not: the duplicate guard returned before the bump, so a
duplicate-spamming run burned model turns without advancing the counter (394 of 570
calls on a measured run), and the graph recursion limit ended the run instead of the
budget. That breaks the M/P match, because the arms spam duplicates at different rates.
"""
import os
os.environ["GENCYBER_STEP_BUDGET"] = "100"
from langchain_core.runnables import RunnableLambda
from application.langgraph.models import deep_generative_workflow as d

SID = "__budget__"
noop = RunnableLambda(lambda s: {"script_output": "out"})
ex, wr, sg = d._make_tools(noop, noop, noop, SID)
d._reset_step_budget(SID)

CMD = "ls -la"
# _MAX_IDENTICAL_RUNS identical calls, then more that the duplicate guard blocks.
msgs = []
calls = d._MAX_IDENTICAL_RUNS + 5
for i in range(calls):
    st = {"session_id": SID, "messages": list(msgs), "execution_evidence": ""}
    out = ex.invoke({"name": "execute_script", "type": "tool_call",
                     "id": f"tc{i}", "tool_call_id": f"tc{i}",
                     "args": {"command": CMD, "state": st}})
    new = (getattr(out, "update", {}) or {}).get("messages") or []
    # replay history so _count_prior_executions sees the repeats
    from langchain_core.messages import AIMessage
    msgs.append(AIMessage(content="", tool_calls=[
        {"name": "execute_script", "args": {"command": CMD}, "id": f"tc{i}"}]))
    msgs.extend(new)

used = d._steps_used(SID)
print(f"calls made      = {calls}")
print(f"budget charged  = {used}")
ok = used == calls
print(("PASS" if ok else "FAIL") + f": every call charged (expected {calls}, got {used})")
raise SystemExit(0 if ok else 1)
