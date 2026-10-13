"""Finite-automaton demonstration: Aho-Corasick DFA matching several query terms at once."""
from collections import deque


def build_dfa(patterns):
    """Return (goto, output) of a complete DFA; goto[state] maps char -> next state."""
    goto, fail, out = [{}], [0], [set()]
    for p in patterns:
        s = 0
        for ch in p:
            if ch not in goto[s]:
                goto.append({}); fail.append(0); out.append(set())
                goto[s][ch] = len(goto) - 1
            s = goto[s][ch]
        out[s].add(p)
    queue = deque(goto[0].values())
    while queue:
        r = queue.popleft()
        for ch, s in goto[r].items():
            queue.append(s)
            f = fail[r]
            while f and ch not in goto[f]:
                f = fail[f]
            fail[s] = goto[f].get(ch, 0)
            out[s] |= out[fail[s]]
    return goto, fail, out


def match(patterns, text, trace_limit=60):
    patterns = sorted({p.lower() for p in patterns if p and p.strip()})
    if not patterns:
        raise ValueError("Provide at least one term.")
    text = str(text or "").lower()
    goto, fail, out = build_dfa(patterns)
    state, matches, trace = 0, [], []
    for i, ch in enumerate(text):
        while state and ch not in goto[state]:
            state = fail[state]
        state = goto[state].get(ch, 0)
        if len(trace) < trace_limit:
            trace.append({"pos": i, "char": ch, "state": state})
        for p in out[state]:
            matches.append({"term": p, "start": i - len(p) + 1, "end": i + 1})
    matches.sort(key=lambda m: (m["start"], m["term"]))
    return {"states": len(goto), "patterns": patterns, "matches": matches, "trace": trace,
            "trace_truncated": len(text) > trace_limit}
