"""Throwaway: read the newest Langfuse trace and summarize Phase 4 skill retrieval.

The experiment harness does NOT tag session_id on traces, so we pick the most
recent trace (or the newest whose name contains an optional substring filter in
argv[1]). Uses api.trace.get(trace_id) because the v2 observations.get_many
endpoint returns 404 in this Langfuse deployment.

Surfaces: skill-section-read / skill-search / memory-fold / memory-recall EVENTs
and the GENERATION input-char distribution (proxy for prompt bloat).
"""
import sys

from infrastructure.observability import langfuse_tracer

name_filter = sys.argv[1] if len(sys.argv) > 1 else ""
client = langfuse_tracer.client()
assert client is not None, "Langfuse client unavailable"
api = client.api

traces = api.trace.list(limit=25).data
if name_filter:
    traces = [t for t in traces if name_filter in (t.name or "")]
print(f"candidate traces (filter={name_filter!r}): {len(traces)}")
if not traces:
    sys.exit(0)

t = traces[0]
print(f"\n=== newest trace {t.id}  name={t.name}  ts={t.timestamp} ===")
full = api.trace.get(t.id)
obs = full.observations or []

by_type = {}
for o in obs:
    by_type[o.type] = by_type.get(o.type, 0) + 1
print("observation counts by type:", by_type)

names = {}
for o in obs:
    names[o.name] = names.get(o.name, 0) + 1

for ev in ("skill-section-read", "skill-search", "memory-fold", "memory-recall"):
    matches = [o for o in obs if o.name == ev]
    print(f"\n{ev}: {len(matches)}")
    for o in matches[:25]:
        print("   ", o.metadata)

# GENERATION input sizes
gens = [o for o in obs if o.type == "GENERATION"]
sizes = []
for g in gens:
    try:
        n = len(str(g.input)) if g.input is not None else 0
    except Exception:
        n = -1
    sizes.append(n)
if sizes:
    ss = sorted(sizes)
    print(f"\nGENERATION spans: {len(gens)}")
    print(f"  input-char sizes: min={min(sizes)} median={ss[len(ss)//2]} max={max(sizes)}")
