# main.py
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from model_manager import SmartModelManager
from registry import PluginRegistry
from config_loader import PromptManager
from state import MemoryManager
from resource_manager import ResourceManager
from tools import get_builtin_tools
from graph_builder import build_graph
from cli import run_cli

def main():
    rm = ResourceManager(interval=5.0); rm.start()
    pm = PromptManager()
    pm.set_global("app_name", "Multi-Agent CLI")
    pm.set_global("current_date", datetime.now().strftime("%Y-%m-%d"))
    mm = SmartModelManager(resource_manager=rm)
    mem = MemoryManager(max_messages=50, compression_threshold=4)
    mem.set_summary_llm(mm.get_llm())
    reg = PluginRegistry(); reg._mm = mm
    for tool in get_builtin_tools(): reg.register_tool(tool)
    cp = MemorySaver()
    app = build_graph(reg, mm, cp)
    s = rm.status()
    gpu = f"GPU: {s[\'gpu\'][\'name\']}" if s.get("gpu") else "GPU: None"
    print(f"\n[System] {gpu} | CPU: {s[\'cpu\'][\'available_mb\']:.0f}MB free")
    print("[System] AlphaEvolve + AutoResearch enabled")
    run_cli(reg, mm, app, mem, pm, rm, cp); rm.stop()

if __name__ == "__main__":
    main()
