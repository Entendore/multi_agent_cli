# config_loader.py
import yaml, re
from typing import Dict, List, Optional, Any
from langchain_core.tools import StructuredTool

class PromptTemplate:
    def __init__(self, name, content):
        self.name = name; self.variables = re.findall(r'\{\{(\w+)\}\}', content)
        self.versions = {"1.0": content}; self.current = "1.0"
    def add_version(self, v, c): self.versions[v] = c
    def set_version(self, v):
        if v in self.versions: self.current = v
    def render(self, variables=None, version=None):
        c = self.versions.get(version or self.current, "")
        if variables:
            for k, v in variables.items(): c = c.replace(f"{{{{{k}}}}}", str(v))
        return c

class PromptManager:
    def __init__(self): self.templates = {}; self.globals = {}
    def register(self, name, content):
        pt = PromptTemplate(name, content); self.templates[name] = pt; return pt
    def render(self, name, variables=None, version=None):
        pt = self.templates.get(name)
        if not pt: return f"[Missing: {name}]"
        return pt.render({**self.globals, **(variables or {})}, version)
    def set_global(self, k, v): self.globals[k] = v

class ConfigLoader:
    @staticmethod
    def load(filepath, registry, model_manager, prompt_manager=None):
        try:
            with open(filepath) as f: data = yaml.safe_load(f)
        except Exception as e:
            print(f"[Error] Cannot read {filepath}: {e}"); return False
        errors = []
        if "model_profiles" in data:
            try: model_manager.load_profiles(data["model_profiles"], data.get("defaults", {}).get("model_profile", "default"))
            except Exception as e: errors.append(f"Profiles: {e}")
        for tc in data.get("tools", data.get("custom_tools", [])):
            try:
                t = ConfigLoader._make_tool(tc)
                if t: registry.register_tool(t)
            except Exception as e: errors.append(f"Tool: {e}")
        pm = prompt_manager
        if pm and "prompt_templates" in data:
            try:
                for n, v in data["prompt_templates"].items():
                    try:
                        if isinstance(v, str): pm.register(n, v)
                        elif isinstance(v, dict):
                            pt = pm.register(n, v["content"])
                            for ver, vd in v.get("versions", {}).items(): pt.add_version(ver, vd["content"])
                    except Exception as e: errors.append(f"Template {n}: {e}")
            except Exception as e: errors.append(f"Templates: {e}")
        if pm and "prompt_globals" in data:
            try: pm.globals.update(data["prompt_globals"])
            except Exception as e: errors.append(f"Globals: {e}")
        if "agents" in data:
            try:
                for a in data["agents"]:
                    try:
                        blocks = a.get("prompt_blocks", [])
                        fp = "\n".join(pm.render(b.replace("$", "")) if b.startswith("$") else b for b in blocks)
                        registry.register_agent(name=a["name"], system_prompt=fp, tool_names=a.get("tools", []), model_profile=a.get("model_profile"), llm_overrides=a.get("llm_overrides"), metadata=a.get("metadata"))
                    except Exception as e: errors.append(f"Agent: {e}")
            except Exception as e: errors.append(f"Agents: {e}")
        if "evolution" in data:
            try: registry._evo_cfg = data["evolution"]
            except Exception as e: errors.append(f"Evolution: {e}")
        if "research" in data:
            try: registry._research_cfg = data["research"]
            except Exception as e: errors.append(f"Research: {e}")
        if "workflow" in data:
            try: registry._workflow_cfg = data["workflow"]
            except Exception as e: errors.append(f"Workflow: {e}")
        print(f"[System] Configuration loaded from {filepath}")
        if errors:
            for err in errors: print(f"[Warning] {err}")
        return True

    @staticmethod
    def _make_tool(conf):
        t = conf.get("type")
        if t == "python":
            scope = {}
            try:
                exec(conf.get("code", ""), {}, scope)
                fn = scope.get("run_tool")
                if fn: return StructuredTool.from_function(func=fn, name=conf["name"], description=conf.get("description", ""))
            except Exception as e: print(f"[Error] Tool compile: {e}")
        elif t == "duckduckgo":
            try:
                from langchain_community.tools import DuckDuckGoSearchRun
                return DuckDuckGoSearchRun(name=conf["name"])
            except ImportError: print("[Error] duckduckgo not installed")
        return None
