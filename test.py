#!/usr/bin/env python3
# test.py
import subprocess, sys, os, time, threading
from queue import Queue, Empty
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Result:
    name: str; passed: bool; desc: str; detail: str = ""

class Tester:
    def __init__(self):
        self.q = Queue(); self.lines = []; self.proc = None; self._mark = 0; self._buf = ""
    def _reader(self):
        try:
            while True:
                c = self.proc.stdout.read(1)
                if not c:
                    if self._buf: self.lines.append(self._buf); self.q.put(self._buf)
                    break
                if c == "\n": self.lines.append(self._buf); self.q.put(self._buf); self._buf = ""
                elif c == "\r": continue
                else: self._buf += c
        except:
            if self._buf: self.lines.append(self._buf); self.q.put(self._buf)
    def wait_for(self, p, t=30):
        t0 = time.time()
        while time.time() - t0 < t:
            try:
                l = self.q.get(timeout=0.2)
                if p.lower() in l.lower(): return True, l
            except Empty: pass
            if p.lower() in self._buf.lower(): return True, self._buf
        return False, None
    def wait_prompt(self, t=30): return self.wait_for("User >>", t)
    def send(self, cmd, wait=True, timeout=180):
        print(f"\n  [CMD] >>> {cmd if cmd else \'(empty)\'}", flush=True)
        try: self.proc.stdin.write(cmd + "\n"); self.proc.stdin.flush()
        except Exception as e: return False, str(e)
        if wait: return self.wait_prompt(timeout)
        time.sleep(0.3); return True, None
    def set_mark(self): self._mark = len(self.lines)
    def has(self, p):
        t = "\n".join(self.lines)
        if self._buf: t += "\n" + self._buf
        return p.lower() in t.lower()
    def has_after(self, p):
        ls = self.lines[self._mark:]
        if self._buf: ls = ls + [self._buf]
        return any(p.lower() in l.lower() for l in ls)
    def drain(self):
        while not self.q.empty():
            try: self.q.get_nowait()
            except Empty: pass
    def start(self):
        env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
        try:
            self.proc = subprocess.Popen([sys.executable, "main.py"], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=0,
                universal_newlines=True, env=env, cwd=os.path.dirname(os.path.abspath(__file__)) or ".")
        except Exception as e: print(f"  [FATAL] {e}"); return False
        threading.Thread(target=self._reader, daemon=True).start()
        return self.wait_prompt(60)[0]
    def stop(self):
        if self.proc:
            try: self.proc.stdin.write("/exit\n"); self.proc.stdin.flush(); self.proc.wait(timeout=10)
            except:
                try: self.proc.kill()
                except: pass

TEST_YAML = """defaults:
  model_profile: qwen_cpu
model_profiles:
  qwen_cpu:
    model_name: "qwen2.5:0.5b"
    temperature: 0.0
prompt_templates:
  math_specialist: |
    You are a math specialist. Solve precisely. Keep responses very short.
  general_assistant: |
    You are a helpful assistant. Keep responses under 50 words.
evolution:
  num_islands: 2
  pop_per_island: 3
  max_generations: 2
  eval_timeout: 3.0
  max_archive_size: 8
research:
  critique_rounds: 1
workflow:
  pattern: supervisor
agents:
  - name: math_agent
    prompt_blocks: ["$math_specialist"]
    tools: [calculator]
    model_profile: qwen_cpu
    metadata: {role: mathematics}
  - name: general_agent
    prompt_blocks: ["$general_assistant"]
    tools: []
    model_profile: qwen_cpu
    metadata: {role: general}
"""

def check_prereqs():
    print("Checking prerequisites...")
    print(f"  Python: {sys.version.split()[0]}")
    try:
        r = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0: print(f"  Ollama: {r.stdout.strip()}")
        else: print("  Ollama: not running"); return False
    except FileNotFoundError: print("  Ollama: not installed"); return False
    try:
        import urllib.request; urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        print("  Server: running on :11434")
    except: print("  Server: not responding"); return False
    r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
    if "qwen2.5" in r.stdout.lower() or "qwen3" in r.stdout.lower(): print("  Qwen model: found")
    else:
        print("  Pulling qwen2.5:0.5b ...")
        p = subprocess.run(["ollama", "pull", "qwen2.5:0.5b"], capture_output=True, text=True, timeout=180)
        if p.returncode != 0: print(f"  Pull failed: {p.stderr[:80]}"); return False
        print("  Qwen model: pulled")
    for pkg in ["langchain_core", "langchain_openai", "langgraph", "yaml"]:
        try: __import__(pkg)
        except ImportError: print(f"  {pkg}: MISSING"); return False
    print("  Packages: OK")
    with open("_test_agents.yaml", "w") as f: f.write(TEST_YAML)
    print("  _test_agents.yaml: created\n"); return True

def run_tests():
    results = []; t = Tester()
    def rec(n, p, d, dt=""):
        results.append(Result(n, p, d, dt))
        print(f"    [{\'PASS\' if p else \'FAIL\'}] {n}: {d}")
        if dt and not p: print(f"          -> {dt}")
    print("\n[1] Startup")
    ok = t.start()
    rec("startup", ok, "starts", "timeout" if not ok else "")
    rec("features", t.has("AlphaEvolve"), "shows AlphaEvolve")
    print("\n[2] Help")
    t.drain(); t.send("/help", wait=False); time.sleep(0.5)
    for kw in ["/load", "/mode", "/evolve", "/research", "/workflow", "/exit"]: rec(f"help_{kw}", t.has(kw), f"shows {kw}")
    print("\n[3] Load")
    t.drain(); t.send("/load _test_agents.yaml", timeout=30)
    rec("load", t.has("Configuration loaded"), "loads")
    print("\n[4] Profiles")
    t.drain(); t.send("/profiles", wait=False); time.sleep(0.5)
    rec("prof", t.has("qwen_cpu"), "shows")
    print("\n[5] List")
    t.drain(); t.send("/list", wait=False); time.sleep(0.5)
    rec("agents", t.has("math_agent"), "agents"); rec("tools", t.has("calculator"), "tools")
    print("\n[6] Mode")
    t.drain(); t.send("/mode qwen_cpu"); rec("mode", t.has("Switched to"), "switches")
    print("\n[7] Invalid")
    t.drain(); t.send("/mode bad"); rec("invalid", t.has("not found"), "error")
    print("\n[8] Unknown")
    t.drain(); t.send("/foobar", wait=False); time.sleep(0.5)
    rec("unknown", t.has("Unknown"), "error")
    print("\n[9] Resources")
    t.drain(); t.send("/resources", wait=False); time.sleep(0.5)
    rec("gpu", t.has("GPU"), "GPU"); rec("cpu", t.has("CPU"), "CPU")
    print("\n[10] Suggest")
    t.drain(); t.send("/suggest", wait=False); time.sleep(0.5)
    rec("suggest", t.has("Suggested"), "suggests")
    print("\n[11] Workflow")
    t.drain(); t.send("/workflow", wait=False); time.sleep(0.5)
    rec("pattern", t.has("supervisor"), "pattern"); rec("evo", t.has("Evolution"), "evo"); rec("res", t.has("Research"), "res")
    print("\n[12] Mem empty")
    t.drain(); t.send("/memory", wait=False); time.sleep(0.5)
    rec("empty", t.has("No messages"), "empty")
    print("\n[13] Chat math")
    t.drain(); t.set_mark(); t.send("What is 2+2?", timeout=90)
    rec("math_a", t.has_after("[math_agent]"), "agent"); rec("math_4", t.has_after("4") or t.has_after("[Tool]"), "4")
    print("\n[14] Chat general")
    t.drain(); t.set_mark(); t.send("What color is the sky?", timeout=90)
    rec("gen_a", t.has_after("[general_agent]"), "agent"); rec("blue", t.has_after("blue"), "blue")
    print("\n[15] Multi-turn")
    t.drain(); t.set_mark(); t.send("What is 3+3?", timeout=90); time.sleep(1); t.send("And 5+5?", timeout=90)
    rec("m6", t.has_after("6") or t.has_after("six"), "6"); rec("m10", t.has_after("10") or t.has_after("ten"), "10")
    print("\n[16] Mem after")
    t.drain(); t.send("/memory", wait=False); time.sleep(0.5)
    rec("msgs", t.has("Messages:"), "count")
    print("\n[17] Compress")
    t.drain(); t.send("/compress", wait=False); time.sleep(3)
    rec("comp", t.has("Compressed") or t.has("Nothing"), "compress")
    print("\n[18] AlphaEvolve")
    t.drain(); t.set_mark()
    t.send("/evolve Write a Python function to check if a number is prime", timeout=240)
    rec("evo_start", t.has_after("AlphaEvolve"), "starts")
    rec("evo_complete", t.has_after("Complete"), "completes")
    rec("evo_islands", t.has_after("Islands:"), "islands")
    rec("evo_evals", t.has_after("evaluations"), "evals")
    rec("evo_fitness", t.has_after("fitness"), "fitness")
    rec("evo_tests", t.has_after("tests pass"), "tests pass")
    rec("evo_archive", t.has_after("Archive:"), "archive")
    rec("evo_code", t.has_after("def solve"), "code fn")
    rec("evo_md", t.has_after("```python"), "md block")
    print("\n[19] AutoResearch")
    t.drain(); t.set_mark()
    t.send("/research Does sleep improve memory?", timeout=300)
    rec("res_start", t.has_after("AutoResearch"), "starts")
    rec("res_complete", t.has_after("Complete"), "completes")
    rec("res_phases", t.has_after("Phases"), "phases")
    rec("res_conf", t.has_after("Confidence"), "confidence")
    rec("res_report", t.has_after("Report"), "report")
    print("\n[20] Empty")
    t.drain(); t.send("", wait=False); time.sleep(1)
    ok2 = t.send("/help", timeout=15)[0]
    rec("empty", ok2, "no crash")
    print("\n[21] Exit")
    t.send("/exit", wait=False); time.sleep(2)
    rec("exit", t.proc.poll() is not None, "exits")
    t.stop(); return results

def summary(results):
    print("\n" + "=" * 60); print("  SUMMARY"); print("=" * 60)
    cats = {}
    for r in results:
        if any(x in r.name for x in ["startup", "features", "help"]): c = "UI"
        elif any(x in r.name for x in ["load", "prof", "mode", "invalid", "unknown"]): c = "Config"
        elif "list" in r.name: c = "Registry"
        elif any(x in r.name for x in ["gpu", "cpu", "suggest", "pattern", "evo_cfg", "res_cfg"]): c = "Config"
        elif any(x in r.name for x in ["mem", "compress"]): c = "Memory"
        elif any(x in r.name for x in ["math", "gen_", "blue", "multi", "m6", "m10", "empty"]): c = "Chat"
        elif "evo" in r.name: c = "AlphaEvolve"
        elif "res_" in r.name and r.name not in ("res_cfg", "res_gpu", "res_cpu"): c = "AutoResearch"
        else: c = "Other"
        cats.setdefault(c, [0, 0, []])
        cats[c][0] += int(r.passed); cats[c][1] += 1; cats[c][2].append(r)
    for cat, (p, tot, items) in cats.items():
        icon = "PASS" if p == tot else "FAIL"
        print(f"\n  [{icon}] {cat}: {p}/{tot}")
        for i in items: print(f"    {\'+\' if i.passed else \'-\'} {i.name}")
    passed = sum(r.passed for r in results); total = len(results)
    print(f"\n  TOTAL: {passed}/{total}")
    if passed < total:
        print("\n  FAILED:")
        for r in results:
            if not r.passed: print(f"    - {r.name}: {r.detail or r.desc}")
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Test Suite — AlphaEvolve (code) + AutoResearch")
    print("=" * 60 + "\n")
    if not check_prereqs(): print("\nPrerequisites NOT met."); sys.exit(1)
    results = run_tests()
    ok = summary(results)
    try: os.remove("_test_agents.yaml")
    except: pass
    sys.exit(0 if ok else 1)
