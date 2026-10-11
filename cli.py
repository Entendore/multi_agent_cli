# cli.py
import sys, time, threading
from queue import Queue
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def print_help():
    print("\n--- Commands ---")
    print("  <msg>         : Chat")
    print("  /load <f>     : Load YAML")
    print("  /mode <p>     : Switch Profile")
    print("  /profiles     : List profiles")
    print("  /list         : List Agents & Tools")
    print("  /resources    : GPU/CPU")
    print("  /suggest      : Suggest model")
    print("  /auto         : Auto-select")
    print("  /memory       : Memory stats")
    print("  /compress     : Compress")
    print("  /workflow     : Config")
    print("  /evolve <t>   : AlphaEvolve")
    print("  /research <t> : AutoResearch")
    print("  /help         : Help")
    print("  /exit         : Quit")
    print("----------------")

def _stream(app, inp, cfg, q):
    try:
        for ev in app.stream({"messages": [HumanMessage(content=inp)]}, cfg, stream_mode="updates"):
            for n, u in ev.items():
                if "messages" not in u: continue
                for m in u["messages"]:
                    if isinstance(m, ToolMessage): q.put(("tool", str(m.content)[:300]))
                    elif isinstance(m, AIMessage) and m.content: q.put(("agent", n, m.content))
    except Exception as e: q.put(("error", str(e)))
    finally: q.put(("done",))

def _run_special(app, arg, label, tout):
    print(f"\n[{label}] Starting: {arg[:70]}...")
    cfg = {"configurable": {"thread_id": f"{label}-{time.time()}"}}
    q = Queue(); threading.Thread(target=_stream, args=(app, arg, cfg, q), daemon=True).start()
    while True:
        try:
            i = q.get(timeout=tout)
            if i[0] == "done": break
            elif i[0] == "agent": print(f"\n{i[2]}")
            elif i[0] == "error": print(f"\n[Error] {i[1]}")
        except: print(f"\n[Error] {label} timeout"); break

def run_cli(reg, mm, app, mem, pm, rm, cp):
    config = {"configurable": {"thread_id": "default-session"}}
    print("\n" + "=" * 60)
    print("  Multi-Agent CLI — AlphaEvolve + AutoResearch")
    print("=" * 60)
    print("Ready. Type \'/load agents.yaml\' to start.")
    print_help()
    threading.Thread(target=mm.prewarm, daemon=True).start()

    def rebuild():
        nonlocal app, config
        from graph_builder import build_graph
        reg._mm = mm; app = build_graph(reg, mm, cp)
        config = {"configurable": {"thread_id": f"session-{time.time()}"}}

    while True:
        sys.stdout.write("\nUser >> "); sys.stdout.flush()
        try: ui = sys.stdin.readline()
        except KeyboardInterrupt: break
        if not ui: break
        ui = ui.rstrip("\n")
        if not ui.strip(): continue

        if ui.startswith("/"):
            p = ui.split(maxsplit=1); cmd, arg = p[0].lower(), p[1] if len(p) > 1 else ""
            if cmd == "/exit": print("Goodbye!"); break
            elif cmd == "/help": print_help()
            elif cmd == "/load" and arg:
                from config_loader import ConfigLoader
                if ConfigLoader.load(arg, reg, mm, pm): rebuild()
            elif cmd == "/mode" and arg:
                if mm.set_mode(arg): rebuild()
            elif cmd == "/evolve" and arg:
                from graph_builder import build_graph
                reg._mm = mm; _run_special(build_graph(reg, mm, cp, pattern="evolve"), arg, "AlphaEvolve", 180)
            elif cmd == "/research" and arg:
                from graph_builder import build_graph
                reg._mm = mm; _run_special(build_graph(reg, mm, cp, pattern="research"), arg, "AutoResearch", 300)
            elif cmd == "/profiles":
                print("\n--- Profiles ---")
                for k, v in mm.profiles.items():
                    m = " <--" if k == mm.current_profile_name else ""
                    print(f"  {k}: {v.get(\'model_name\')} temp={v.get(\'temperature\', 0)}{m}")
            elif cmd == "/list":
                print("\n--- Agents ---")
                for k, v in reg.agents.items(): print(f"  {k}: {v[\'tool_names\']}")
                print("--- Tools ---")
                for k, v in reg.tools.items(): print(f"  {k}: {v.description[:50] if v.description else \'\'}")
            elif cmd == "/resources":
                s = rm.status()
                print(f"\n  GPU: {\'Yes\' if s[\'has_gpu\'] else \'No\'}")
                if s.get("gpu"): print(f"  {s[\'gpu\'][\'name\']}: {s[\'gpu\'][\'mem_used_mb\']}/{s[\'gpu\'][\'mem_total_mb\']}MB")
                print(f"  CPU: {s[\'cpu\'][\'mem_used_mb\']}/{s[\'cpu\'][\'mem_total_mb\']}MB")
            elif cmd == "/suggest":
                sug = rm.suggest_model(); print(f"\n  Suggested: {sug}")
            elif cmd == "/auto":
                print(f"\n  Auto: {mm.auto_select()}"); rebuild()
            elif cmd == "/memory":
                try:
                    st = app.get_state(config); msgs = st.values.get("messages", []) if st.values else []
                except: msgs = []
                if msgs:
                    stats = mem.stats(msgs, mm.current_model_name)
                    print(f"\n  Msgs: {stats[\'message_count\']} | Tok: {stats[\'used_tokens\']}/{stats[\'total_tokens\']}")
                    if stats["warning"]: print("  Warning: high usage")
                else: print("\n  No messages.")
            elif cmd == "/compress":
                try:
                    st = app.get_state(config); msgs = st.values.get("messages", []) if st.values else []
                except: msgs = []
                if msgs:
                    old = len(msgs); _, ok = mem.compress(msgs, mm.get_llm())
                    if ok: config["configurable"]["thread_id"] = f"comp-{time.time()}"; print(f"\n  Compressed: {old}")
                    else: print("\n  Nothing to compress")
                else: print("\n  Nothing to compress")
            elif cmd == "/workflow":
                wc = getattr(reg, "_workflow_cfg", None)
                print(f"\n  Pattern: {wc.get(\'pattern\', \'supervisor\') if isinstance(wc, dict) else \'supervisor\'}")
                ec = getattr(reg, "_evo_cfg", None)
                if ec: print(f"  Evolution: {ec.get(\'num_islands\', 2)} islands, {ec.get(\'max_generations\', 2)} gen")
                rc = getattr(reg, "_research_cfg", None)
                if rc: print(f"  Research: {rc.get(\'critique_rounds\', 1)} critique rounds")
            else: print(f"  Unknown: {cmd}")
            continue

        q = Queue()
        threading.Thread(target=_stream, args=(app, ui, config, q), daemon=True).start()
        while True:
            try:
                i = q.get(timeout=120)
                if i[0] == "done": break
                elif i[0] == "tool": print(f"\n[Tool] {i[1]}")
                elif i[0] == "agent": print(f"\n[{i[1]}]\n  {i[2]}")
                elif i[0] == "error": print(f"\n[Error] {i[1]}")
            except: print("\n[Error] Timeout"); break
