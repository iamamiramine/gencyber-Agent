"""Throwaway: read the latest trace for a session and summarize memory effectiveness.

Counts GENERATION spans, surfaces memory-fold / memory-recall EVENT observations
(with metadata), and reports per-generation input sizes so we can see whether the
folded-context changes bounded the prompt growth.
"""
import sys

from infrastructure.observability import langfuse_tracer

session_id = sys.argv[1] if len(sys.argv) > 1 else "exp-mem-phase-coinslot-2016q-msc-coinslot"
client = langfuse_tracer.client()
assert client is not None, "Langfuse client unavailable"

api = client.api
traces = api.trace.list(session_id=session_id, limit=10).data
print(f"traces for session {session_id!r}: {len(traces)}")
if not traces:
    sys.exit(0)

for t in traces:
    print(f"\n=== trace {t.id}  name={t.name}  ts={t.timestamp} ===")
    obs = []
    page = 1
    while True:
        resp = api.observations.get_many(trace_id=t.id, limit=100, page=page)
        obs.extend(resp.data)
        if len(resp.data) < 100:
            break
        page += 1
    by_type = {}
    for o in obs:
        by_type[o.type] = by_type.get(o.type, 0) + 1
    print("observation counts by type:", by_type)

    # memory events
    mem = [o for o in obs if o.name in ("memory-fold", "memory-recall", "route-after-generation")]
    folds = [o for o in obs if o.name == "memory-fold"]
    recalls = [o for o in obs if o.name == "memory-recall"]
    print(f"memory-fold events: {len(folds)} | memory-recall events: {len(recalls)}")
    for o in folds[:20]:
        print("  FOLD ", o.metadata)
    for o in recalls[:20]:
        print("  RECALL", o.metadata)

    # generation span input sizes (proxy for context growth)
    gens = [o for o in obs if o.type == "GENERATION"]
    sizes = []
    for g in gens:
        try:
            n = len(str(g.input)) if g.input is not None else 0
        except Exception:
            n = -1
        sizes.append(n)
    if sizes:
        sizes_sorted = sorted(sizes)
        print(f"GENERATION spans: {len(gens)}")
        print(f"  input-char sizes: min={min(sizes)} median={sizes_sorted[len(sizes)//2]} max={max(sizes)}")
