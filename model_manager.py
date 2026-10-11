# model_manager.py
import json, threading
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from resource_manager import ResourceManager

class SmartModelManager:
    def __init__(self, resource_manager=None):
        self.profiles = {}; self.current_profile_name = "default"
        self.rm = resource_manager or ResourceManager(); self.rm.start()
        self._cache = {}; self._cache_lock = threading.Lock(); self._prewarmed = False
        self.load_profiles({"default": {"model_name": "qwen2.5:0.5b", "temperature": 0.0}}, "default")

    def load_profiles(self, profiles, default_name):
        self.profiles = profiles
        self.current_profile_name = default_name if default_name in profiles else "default"
        with self._cache_lock: self._cache.clear(); self._prewarmed = False
        print(f"[ModelManager] Loaded {len(profiles)} profiles. Active: {self.current_profile_name}")

    def set_mode(self, mode):
        if mode not in self.profiles: print(f"[ModelManager] Error: \'{mode}\' not found."); return False
        m = self.profiles[mode].get("model_name", "")
        c = self.rm.can_load(m)
        if not c["can_load"]: print(f"[ModelManager] Cannot load {m}"); return False
        self.current_profile_name = mode; print(f"[ModelManager] Switched to {mode}"); return True

    def _create_llm(self, pname, overrides=None):
        if pname not in self.profiles: pname = "default"
        cfg = self.profiles[pname]
        opts = {"temperature": cfg.get("temperature", 0.0), "top_p": cfg.get("top_p", 1.0), "max_tokens": min(cfg.get("max_tokens", 2048), 4096)}
        if overrides:
            for k, v in overrides.items():
                if k in {"temperature","top_p","frequency_penalty","presence_penalty","max_tokens"}: opts[k] = v
        return ChatOpenAI(base_url="http://localhost:11434/v1", api_key="dummy-key", model=cfg.get("model_name", "qwen2.5:0.5b"), **opts)

    def get_llm(self, override_profile=None, llm_overrides=None):
        pname = override_profile or self.current_profile_name
        if pname not in self.profiles: pname = "default"
        key = f"{pname}:{json.dumps(llm_overrides or {}, sort_keys=True)}"
        with self._cache_lock:
            if key in self._cache: return self._cache[key]
        llm = self._create_llm(pname, llm_overrides)
        with self._cache_lock:
            if key not in self._cache: self._cache[key] = llm
            return self._cache[key]

    def prewarm(self):
        if self._prewarmed: return
        def w(pn):
            try: self.get_llm(pn).invoke("hi", max_tokens=1)
            except: pass
        with ThreadPoolExecutor(max_workers=min(len(self.profiles), 4)) as p: list(p.map(w, self.profiles))
        with self._cache_lock: self._prewarmed = True
        print("[ModelManager] LLMs pre-warmed.")

    def get_resource_status(self): return self.rm.status()
    def auto_select(self):
        s = self.rm.suggest_model()
        for pn, c in self.profiles.items():
            if c.get("model_name") == s: self.set_mode(pn); return pn
        return self.current_profile_name

    @property
    def current_model_name(self):
        return self.profiles.get(self.current_profile_name, {}).get("model_name", "qwen2.5:0.5b")
