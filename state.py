# state.py
import threading, time, hashlib, ast
from typing import TypedDict, Annotated, List, Dict, Optional, Tuple, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import add_messages
from concurrent.futures import ThreadPoolExecutor, as_completed

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ═══════════════════════════════════════════════════════════
#  AlphaEvolve — Code-Centric Evolutionary Search
# ═══════════════════════════════════════════════════════════
class EvalResult:
    __slots__ = ("passed", "failed", "errors", "score", "runtime_ms")
    def __init__(self, passed=0, failed=0, errors=0, score=0.0, runtime_ms=0.0):
        self.passed, self.failed, self.errors, self.score, self.runtime_ms = passed, failed, errors, score, runtime_ms

class Solution:
    def __init__(self, code, island=0, gen=0, parent_ids=None):
        self.code, self.island, self.gen = code, island, gen
        self.parent_ids = parent_ids or []
        self.signature = hashlib.sha256(code.encode()).hexdigest()[:12]
        self.eval_result = None
    @property
    def fitness(self): return self.eval_result.score if self.eval_result else 0.0

class ArchiveEntry:
    def __init__(self, solution): self.solution, self.added_at = solution, time.time()

class Archive:
    def __init__(self, max_size=20, min_improvement=0.01):
        self.entries, self.max_size, self.min_improvement = [], max_size, min_improvement
        self._lock = threading.Lock()
    def try_add(self, sol):
        with self._lock:
            if self.entries and sol.fitness < min(e.solution.fitness for e in self.entries) + self.min_improvement: return False
            if any(e.solution.signature == sol.signature for e in self.entries): return False
            self.entries.append(ArchiveEntry(sol))
            if len(self.entries) > self.max_size:
                self.entries.sort(key=lambda e: e.solution.fitness, reverse=True)
                self.entries = self.entries[:self.max_size]
            return True
    def sample(self, count=2):
        with self._lock:
            if len(self.entries) <= count: return [e.solution for e in self.entries]
            return [e.solution for e in random.sample(self.entries, count)]
    def best(self):
        with self._lock: return max(self.entries, key=lambda e: e.solution.fitness).solution if self.entries else None
    def status(self):
        with self._lock:
            sc = [e.solution.fitness for e in self.entries]
            return {"size": len(self.entries), "best_score": max(sc) if sc else 0, "avg_score": sum(sc)/len(sc) if sc else 0}

class Island:
    def __init__(self, iid, scaffold, tests, llm, timeout=5.0, archive=None):
        self.id, self.scaffold, self.test_cases, self.llm = iid, scaffold, tests, llm
        self.eval_timeout, self.archive = timeout, archive
        self.population, self.generation, self.best = [], 0, None
        self.total_evals, self.log = 0, []
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"island_{iid}")

class EvolutionEngine:
    def __init__(self, task, scaffold, tests, llm, num_islands=2, pop=3, max_gen=3, timeout=5.0, mut_rate=0.7, cross_rate=0.2, arch_rate=0.1, max_arch=16):
        self.task, self.scaffold, self.tests, self.llm = task, scaffold, tests, llm
        self.num_islands, self.pop, self.max_gen = num_islands, pop, max_gen
        self.timeout, self.mut_rate, self.cross_rate, self.arch_rate = timeout, mut_rate, cross_rate, arch_rate
        self.archive = Archive(max_size=max_arch)
        self.islands, self.best_overall, self.log = [], None, []
        self._running, self._total_evals, self._start = False, 0, 0

    def _sandbox_exec(self, code):
        import io
        t0 = time.time(); p, f, e = 0, 0, 0
        for tc in self.test_cases:
            inp, exp = tc.get("input", ()), tc.get("expected", None)
            sandbox = f"{self.scaffold}\n# --- Evolved ---\n{code}\n_args={repr(inp)}\n_expected={repr(exp)}\ntry:\n  _r=solve(*_args)\n  print(f\'PASS:\' if _r==_expected else f\'FAIL:\')\nexcept Exception as _e:\n  print(f\'ERROR:\')"
            try:
                out = io.StringIO()
                exec(sandbox, {"__builtins__": __builtins__, "print": lambda *a, **k: out.write(" ".join(str(x) for x in a)+"\n")}, {})
                o = out.getvalue()
                if "PASS" in o: p += 1
                elif "ERROR" in o: e += 1
                else: f += 1
            except: e += 1
        tot = p + f + e
        return EvalResult(p, f, e, (p/tot if tot else 0) + (0.05 if (time.time()-t0)*1000 < self.timeout*1000 else 0), (time.time()-t0)*1000)

    def _llm_gen(self, prompt):
        try:
            r = self.llm.invoke(prompt, max_tokens=512)
            try:
                tree = ast.parse(r.content)
                for n in ast.walk(tree):
                    if isinstance(n, ast.FunctionDef) and n.name == "solve": return ast.unparse(n)
            except: pass
            import re
            m = re.search(r'(def\s+solve\s*\([^)]*\)\s*:\n(?:\s{4,}.+\n)*)', r.content, re.M)
            return m.group(1).strip() if m else "def solve(*a): return None"
        except: return "def solve(*a): return None"

    def _mutate(self, sol, isl):
        hint = ""
        if self.archive.entries:
            s = self.archive.sample(1)
            if s: hint = f"\nDifferent working approach (learn, don\'t copy):\n```python\n{s[0].code}\n```"
        p = f"Task: {self.task}\nCurrent ({sol.eval_result.passed}/{len(self.tests)} pass):\n```python\n{sol.code}\n```{hint}\nImprove. Output ONLY def solve(...)."
        return Solution(self._llm_gen(p), isl.id, isl.generation+1, [sol.signature])

    def _crossover(self, p1, p2, isl):
        p = f"Task: {self.task}\nA ({p1.fitness:.2f}):\n```python\n{p1.code}\n```\nB ({p2.fitness:.2f}):\n```python\n{p2.code}\n```\nCombine into ONE def solve(...)."
        return Solution(self._llm_gen(p), isl.id, max(p1.gen, p2.gen)+1, [p1.signature, p2.signature])

    def _arch_mutate(self, isl):
        s = self.archive.sample(1)
        if not s: return self._mutate(isl.population[0], isl)
        p = f"Task: {self.task}\nAdapt this approach:\n```python\n{s[0].code}\n```\nOutput ONLY def solve(...)."
        return Solution(self._llm_gen(p), isl.id, isl.generation+1, [s[0].signature])

    def _seed(self, isl):
        sols = []
        for i in range(self.pop):
            p = f"Task: {self.task}\nWrite def solve(...). Be creative. Output ONLY the function."
            sols.append(Solution(self._llm_gen(p), isl.id, 0))
        futs = {isl._pool.submit(self._sandbox_exec, s.code): s for s in sols}
        for f in as_completed(futs, timeout=self.timeout*len(self.tests)+10):
            s = futs[f]
            try: s.eval_result = f.result(timeout=30)
            except: s.eval_result = EvalResult(errors=len(self.tests))
            isl.total_evals += 1; self._total_evals += 1; self.archive.try_add(s)
        isl.population = sorted(sols, key=lambda x: x.fitness, reverse=True)
        isl.best = isl.population[0]

    def _evolve_isl(self, isl):
        isl.generation += 1; new = []
        elite = Solution(isl.best.code, isl.id, isl.generation, [isl.best.signature]); elite.eval_result = isl.best.eval_result
        new.append(elite)
        while len(new) < self.pop:
            r = random.random()
            if r < self.cross_rate and len(isl.population) >= 2:
                p1, p2 = self._tournament(isl.population), self._tournament(isl.population)
                new.append(self._crossover(p1, p2, isl))
            elif r < self.cross_rate + self.arch_rate and self.archive.entries:
                new.append(self._arch_mutate(isl))
            else:
                new.append(self._mutate(self._tournament(isl.population), isl))
        futs = {isl._pool.submit(self._sandbox_exec, s.code): s for s in new if s.eval_result is None}
        for f in as_completed(futs, timeout=self.timeout*len(self.tests)+10):
            s = futs[f]
            try: s.eval_result = f.result(timeout=30)
            except: s.eval_result = EvalResult(errors=len(self.tests))
            isl.total_evals += 1; self._total_evals += 1; self.archive.try_add(s)
        isl.population = sorted(new, key=lambda x: x.fitness, reverse=True); isl.best = isl.population[0]
        if not self.best_overall or isl.best.fitness > self.best_overall.fitness: self.best_overall = isl.best

    def _tournament(self, pop, k=2): return max(random.sample(pop, min(k, len(pop))), key=lambda x: x.fitness)

    def run(self):
        self._running, self._start, self._total_evals = True, time.time(), 0
        self.log = [f"[AlphaEvolve] {self.num_islands} islands x {self.pop} pop x {self.max_gen} gen"]
        self.islands = [Island(i, self.scaffold, self.tests, self.llm, self.timeout, self.archive) for i in range(self.num_islands)]
        mp = ThreadPoolExecutor(max_workers=self.num_islands, thread_name_prefix="evo_seed")
        futs = {mp.submit(self._seed, i): i for i in self.islands}
        for f in as_completed(futs, 120):
            try: f.result(timeout=120)
            except: pass
        mp.shutdown(wait=False)
        for _ in range(self.max_gen):
            gp = ThreadPoolExecutor(max_workers=self.num_islands, thread_name_prefix="evo_gen")
            futs = {gp.submit(self._evolve_isl, i): i for i in self.islands}
            for f in as_completed(futs, 180):
                try: f.result(timeout=180)
                except: pass
            gp.shutdown(wait=False)
        for i in self.islands:
            self.log.extend(i.log); i._pool.shutdown(wait=False)
        self.log.append(f"[AlphaEvolve] Done. {self._total_evals} evals. Archive: {len(self.archive.entries)}")
        if self.best_overall: self.log.append(f"[AlphaEvolve] Best fitness: {self.best_overall.fitness:.2f}")
        self._running = False
        return self.best_overall or Solution("def solve(*a): return None")

# ═══════════════════════════════════════════════════════════
#  AutoResearch
# ═══════════════════════════════════════════════════════════
class ResearchPhase:
    HYPOTHESIS="hypothesis"; DESIGN="experiment_design"; EXECUTE="execute"
    ANALYZE="analyze"; SYNTHESIZE="synthesize"; CRITIQUE="critique"
    REPORT="report"; DONE="done"
    ORDER = [HYPOTHESIS, DESIGN, EXECUTE, ANALYZE, SYNTHESIZE, CRITIQUE, REPORT, DONE]

class ResearchEngine:
    def __init__(self, topic, llm, max_rounds=2):
        self.topic, self.llm, self.max_rounds = topic, llm, max_rounds
        self.phase, self.critique_count, self.findings = ResearchPhase.HYPOTHESIS, 0, {}
        self.report, self.confidence, self.log, self._running = "", 0.0, [], False

    def _run_phase(self, phase, ctx=""):
        ps = {ResearchPhase.HYPOTHESIS: f"Topic: {self.topic}\nGenerate 2-3 testable hypotheses. Format H1: ... H2: ...",
              ResearchPhase.DESIGN: f"Topic: {self.topic}\nHypotheses:\n{ctx}\nDesign experiment.",
              ResearchPhase.EXECUTE: f"Topic: {self.topic}\nPlan:\n{ctx}\nSimulate running it.",
              ResearchPhase.ANALYZE: f"Topic: {self.topic}\nObservations:\n{ctx}\nAnalyze results vs hypotheses.",
              ResearchPhase.SYNTHESIZE: f"Topic: {self.topic}\nFindings:\n{ctx}\nSynthesize insights.",
              ResearchPhase.CRITIQUE: f"Topic: {self.topic}\nSynthesis:\n{ctx}\nCritique weaknesses. Rate 0-10.",
              ResearchPhase.REPORT: f"Topic: {self.topic}\nFindings:\n{ctx}\nWrite report: ## Abstract ## Methods ## Results ## Conclusion"}
        try: return self.llm.invoke(ps.get(phase, ""), max_tokens=1024).content
        except Exception as e: return f"[Error: {e}]"

    def run(self):
        self._running, self.phase, self.critique_count, self.findings = True, ResearchPhase.HYPOTHESIS, 0, {}
        self.log = [f"[Research] Topic: {self.topic[:80]}"]
        for phase in ResearchPhase.ORDER:
            if phase == ResearchPhase.DONE: self.phase = phase; break
            self.phase = phase; self.log.append(f"[Research] {phase}")
            ctx = "\n---\n".join(f"{k}: {v}" for k,v in self.findings.items())
            self.findings[phase] = self._run_phase(phase, ctx)
            if phase == ResearchPhase.CRITIQUE:
                self.critique_count += 1
                try:
                    for l in self.findings[phase].split("\n"):
                        if "confidence" in l.lower() or "rating" in l.lower():
                            d = "".join(c for c in l if c.isdigit() or c==".")
                            if d: self.confidence = float(d[:3]); break
                except: self.confidence = 5.0
                if self.confidence < 6.0 and self.critique_count < self.max_rounds:
                    self.log.append(f"[Research] Confidence {self.confidence:.0f}/10 - revising")
                    self.phase = ResearchPhase.HYPOTHESIS; continue
        self.phase = ResearchPhase.REPORT
        ctx = "\n---\n".join(f"{k}: {v}" for k,v in self.findings.items())
        self.report = self._run_phase(ResearchPhase.REPORT, ctx)
        self.findings["final_report"] = self.report
        self.log.append(f"[Research] Done. Confidence: {self.confidence:.0f}/10"); self._running = False
        return self.report

# ═══════════════════════════════════════════════════════════
#  MemoryManager
# ═══════════════════════════════════════════════════════════
class MemoryManager:
    CT = 0.25
    CW = {"qwen2.5:0.5b":32768,"qwen2.5:1.5b":32768,"qwen2.5:3b":32768,"qwen2.5:7b":32768,"qwen3:0.6b":32768,"qwen3:1.7b":32768,"qwen3:4b":32768,"qwen3:8b":32768,"llama3:8b":8192,"phi3:mini":4096}
    def __init__(self, max_msgs=50, comp_thresh=4): self.max_msgs, self.comp_thresh, self._llm = max_msgs, comp_thresh, None
    def set_summary_llm(self, l): self._llm = l
    def est_tok(self, m): return int(sum(len(str(x.content)) for x in m)*self.CT)
    def trim(self, m, mt=None): return m[-self.max_msgs:] if not mt else (r:=[] , c:=0, [r.insert(0,x) or c:=c+int(len(str(x.content))*self.CT) for x in reversed(m) if c+int(len(str(x.content))*self.CT)<=mt][0] if False else r)
    def compress(self, m, l=None):
        if len(m) <= self.comp_thresh: return m, False
        old, rec = m[:-self.comp_thresh], m[-self.comp_thresh:]
        s = self._sum(old, l or self._llm)
        return [SystemMessage(content=f"[Previous summary]\n{s}")] + rec, True if s else False
    def _sum(self, m, l):
        c = "\n".join(f"{'User' if isinstance(x,HumanMessage) else 'Asst'}: {str(x.content)[:120]}" for x in m)
        if l:
            try: return l.invoke(f"Summarize bullets:\n{c}\nSummary:").content
            except: pass
        return "Earlier:\n" + "\n".join(f"- {str(x.content)[:80]}" for x in m[:8])
    def stats(self, m, mn):
        t = self.CW.get(mn, 4096); u = self.est_tok(m)
        return {"model":mn,"total_tokens":t,"used_tokens":u,"available_tokens":t-u,"usage_percent":round(u/t*100,1),"message_count":len(m),"warning":u/t>0.8}
