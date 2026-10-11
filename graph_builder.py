# graph_builder.py
from enum import Enum
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, AIMessage
from state import AgentState, EvolutionEngine, ResearchEngine

class ParallelToolNode:
    def __init__(self, tools, timeout=30, max_workers=4):
        self.tools = list(tools); self.by_name = {t.name: t for t in self.tools}
        self.timeout = timeout; self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool_exec")
    def __call__(self, state):
        msgs = state.get("messages", [])
        if not msgs: return {"messages": []}
        last = msgs[-1]
        if not hasattr(last, "tool_calls") or not last.tool_calls: return {"messages": []}
        res, futs = [], {}
        for c in last.tool_calls:
            t = self.by_name.get(c["name"])
            if t: futs[self._pool.submit(t.invoke, c["args"])] = c
        for f in as_completed(futs, timeout=self.timeout):
            c = futs[f]
            try: ct = str(f.result(timeout=max(1, self.timeout - 1)))
            except Exception as e: ct = f"Error: {e}"
            res.append(ToolMessage(content=ct, tool_call_id=c["id"]))
        return {"messages": res}
    def shutdown(self): self._pool.shutdown(wait=False)

class Pattern(Enum):
    SUPERVISOR = "supervisor"; SWARM = "swarm"; PIPELINE = "pipeline"; EVOLVE = "evolve"; RESEARCH = "research"

def build_graph(registry, mm, cp=None, pattern=None):
    agents = list(registry.agents.keys())
    if not agents: return _empty(cp)
    cfg = getattr(registry, "_workflow_cfg", {})
    p = pattern or (cfg.get("pattern", "supervisor") if isinstance(cfg, dict) else "supervisor")
    try: pat = Pattern(p)
    except ValueError: pat = Pattern.SUPERVISOR
    if pat == Pattern.EVOLVE: return _evolve(registry, mm, cp)
    if pat == Pattern.RESEARCH: return _research(registry, mm, cp)
    workers = {}
    with ThreadPoolExecutor(max_workers=max(1, len(agents)), thread_name_prefix="build") as pool:
        futs = {pool.submit(_mk_worker, registry, n, mm): n for n in agents}
        for f in as_completed(futs, timeout=30):
            n = futs[f]
            try: workers[n] = f.result(timeout=30)
            except: workers[n] = _fallback(n)
    return {Pattern.SUPERVISOR: _sup, Pattern.SWARM: _swarm, Pattern.PIPELINE: _pipe}[pat](registry, workers, agents, cfg if isinstance(cfg, dict) else {}, cp)

def _empty(cp):
    w = StateGraph(AgentState); w.add_node("idle", lambda s: {"messages": []})
    w.set_entry_point("idle"); w.add_edge("idle", END); return w.compile(checkpointer=cp)

def _fallback(n):
    w = StateGraph(AgentState); w.add_node(n, lambda s: {"messages": []})
    w.set_entry_point(n); w.add_edge(n, END); return w.compile()

def _mk_worker(reg, n, mm):
    c = reg.agents[n]; t = reg.get_tools_for_agent(n)
    l = mm.get_llm(override_profile=c.get("model_profile"), llm_overrides=c.get("llm_overrides"))
    try: return create_react_agent(l, t, prompt=c["system_prompt"])
    except TypeError: return _manual(l, t, c["system_prompt"])

def _manual(l, t, p):
    w = StateGraph(AgentState)
    ch = (ChatPromptTemplate.from_messages([("system", p), MessagesPlaceholder("messages")]) | (l.bind_tools(t) if t else l))
    def fn(s): return {"messages": [ch.invoke({"messages": s["messages"]})]}
    w.add_node("agent", fn)
    if t:
        w.add_node("tools", ParallelToolNode(t))
        w.add_conditional_edges("agent", lambda s: "tools" if (s["messages"] and hasattr(s["messages"][-1], "tool_calls") and s["messages"][-1].tool_calls) else END, {"tools": "tools", END: END})
        w.add_edge("tools", "agent")
    else: w.add_edge("agent", END)
    w.set_entry_point("agent"); return w.compile()

def _def_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(base_url="http://localhost:11434/v1", api_key="dummy-key", model="qwen2.5:0.5b")

def _sup(reg, w, a, c, cp):
    wg = StateGraph(AgentState)
    for n in a:
        if n in w: wg.add_node(n, w[n])
    mm = getattr(reg, "_mm", None); sl = mm.get_llm() if mm else _def_llm()
    sp = ChatPromptTemplate.from_messages([("system", "Pick one worker. Reply ONLY the name.\nWorkers:\n" + "\n".join(f"- {x}" for x in a) + "\nIf done: FINISH"), MessagesPlaceholder("messages")])
    def sf(s):
        r = (sp | sl).invoke({"messages": s["messages"]}); c = r.content.strip().lower()
        for x in a:
            if x.replace("_", "") in c.replace("_", "").replace(" ", ""): return Command(goto=x)
        return {}
    wg.add_node("supervisor", sf); wg.add_edge("supervisor", END)
    for x in a:
        if x in w: wg.add_edge(x, "supervisor")
    wg.set_entry_point("supervisor"); return wg.compile(checkpointer=cp)

def _swarm(reg, w, a, c, cp):
    wg = StateGraph(AgentState)
    for n in a:
        if n in w: wg.add_node(n, w[n])
    wg.add_node("router", lambda s: Command(goto=a[0])); wg.add_edge("router", a[0])
    def mk(c):
        def r(s):
            if s.get("messages"):
                ct = str(s["messages"][-1].content).lower()
                for x in a:
                    if x.replace("_", "") in ct.replace("_", "").replace(" ", "") and x != c: return Command(goto=x)
            return Command(goto="__end__")
        return r
    for x in a:
        opts = [y for y in a if y != x] + [END]
        wg.add_conditional_edges(x, mk(x), {y: y for y in opts})
    wg.set_entry_point("router"); return wg.compile(checkpointer=cp)

def _pipe(reg, w, a, c, cp):
    wg = StateGraph(AgentState)
    st = c.get("stages", a); v = [s for s in st if s in w] or [x for x in a if x in w][:1]
    for n in v: wg.add_node(n, w[n])
    for i in range(len(v) - 1): wg.add_edge(v[i], v[i + 1])
    wg.add_edge(v[-1], END); wg.set_entry_point(v[0]); return wg.compile(checkpointer=cp)

def _infer_scaffold(task):
    import re
    tl = task.lower(); tc = [{"input": (2,), "expected": True}, {"input": (4,), "expected": False}, {"input": (1,), "expected": False}, {"input": (17,), "expected": True}]
    if "sort" in tl: tc = [{"input": ([3, 1, 2],), "expected": [1, 2, 3]}, {"input": ([],), "expected": []}]
    elif "fib" in tl: tc = [{"input": (0,), "expected": 0}, {"input": (5,), "expected": 5}]
    elif "reverse" in tl: tc = [{"input": ("hello",), "expected": "olleh"}]
    return "", tc

def _evolve(reg, mm, cp):
    wg = StateGraph(AgentState)
    def ev(s):
        msgs = s.get("messages", []); task = msgs[-1].content if msgs else "Write a function"
        m = getattr(reg, "_mm", None)
        if not m: return {"messages": []}
        ec = getattr(reg, "_evo_cfg", {})
        sc, tc = ec.get("scaffold", ""), ec.get("test_cases", [])
        if not sc and not tc: sc, tc = _infer_scaffold(task)
        eng = EvolutionEngine(task, sc, tc, m.get_llm(), ec.get("num_islands", 2), ec.get("pop_per_island", 3), ec.get("max_generations", 2), ec.get("eval_timeout", 3.0))
        best = eng.run()
        for l in eng.log: print(f"  {l}")
        r = best.eval_result
        return {"messages": [AIMessage(content=f"[AlphaEvolve Complete]\nIslands: {eng.num_islands} | Generations: {eng.max_gen}\nEvals: {eng._total_evals}\nArchive: {len(eng.archive.entries)}\nFitness: {best.fitness:.2f} ({r.passed}/{len(eng.test_cases)} pass)\n\n--- Code ---\n```python\n{best.code}\n```")]}
    wg.add_node("evolve", ev); wg.set_entry_point("evolve"); wg.add_edge("evolve", END)
    return wg.compile(checkpointer=cp)

def _research(reg, mm, cp):
    wg = StateGraph(AgentState)
    def rs(s):
        msgs = s.get("messages", []); topic = msgs[-1].content if msgs else "Research topic"
        m = getattr(reg, "_mm", None)
        if not m: return {"messages": []}
        rc = getattr(reg, "_research_cfg", {})
        eng = ResearchEngine(topic, m.get_llm(), rc.get("critique_rounds", 1))
        rep = eng.run()
        for l in eng.log: print(f"  {l}")
        return {"messages": [AIMessage(content=f"[Research Complete]\nPhases: {len(eng.findings)} | Critiques: {eng.critique_count}\nConfidence: {eng.confidence:.0f}/10\n\n--- Report ---\n{rep}")]}
    wg.add_node("research", rs); wg.set_entry_point("research"); wg.add_edge("research", END)
    return wg.compile(checkpointer=cp)
