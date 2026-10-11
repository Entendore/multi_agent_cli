# resource_manager.py
import subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List
from dataclasses import dataclass

@dataclass
class ResourceSnapshot:
    timestamp: float = 0.0; gpu_name: str = ""
    gpu_mem_used: float = 0; gpu_mem_total: float = 0; gpu_util: float = 0
    cpu_mem_used: float = 0; cpu_mem_total: float = 0; cpu_percent: float = 0
    @property
    def gpu_available(self): return max(0, self.gpu_mem_total - self.gpu_mem_used)
    @property
    def cpu_available(self): return max(0, self.cpu_mem_total - self.cpu_mem_used)
    def to_dict(self):
        return {"has_gpu": bool(self.gpu_mem_total),
            "gpu": {"name": self.gpu_name, "mem_used_mb": round(self.gpu_mem_used), "mem_total_mb": round(self.gpu_mem_total), "available_mb": round(self.gpu_available), "util_pct": round(self.gpu_util, 1)} if self.gpu_mem_total else None,
            "cpu": {"mem_used_mb": round(self.cpu_mem_used), "mem_total_mb": round(self.cpu_mem_total), "available_mb": round(self.cpu_available), "pct": round(self.cpu_percent, 1)}}

@dataclass
class ModelSize:
    name: str; gpu_mb: float; cpu_mb: float; can_cpu: bool = True

MODEL_SIZES = {
    "qwen2.5:0.5b": ModelSize("qwen2.5:0.5b", 800, 500),
    "qwen2.5:1.5b": ModelSize("qwen2.5:1.5b", 1500, 1000),
    "qwen2.5:3b": ModelSize("qwen2.5:3b", 2500, 2000),
    "qwen2.5:7b": ModelSize("qwen2.5:7b", 5500, 4500, False),
    "qwen3:0.6b": ModelSize("qwen3:0.6b", 900, 550),
    "qwen3:1.7b": ModelSize("qwen3:1.7b", 1700, 1100),
    "qwen3:4b": ModelSize("qwen3:4b", 3000, 2500),
    "qwen3:8b": ModelSize("qwen3:8b", 6000, 5000, False),
    "llama3:8b": ModelSize("llama3:8b", 6000, 5000, False),
    "phi3:mini": ModelSize("phi3:mini", 2800, 2200),
}

class ResourceManager:
    def __init__(self, interval=5.0):
        self.interval = interval; self.snapshot = ResourceSnapshot()
        self.history = []; self._thread = None; self._running = False
        self.has_gpu = False; self._gpu_checked = False; self._lock = threading.Lock()

    def _detect_gpu(self):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=2)
            return r.returncode == 0 and bool(r.stdout.strip())
        except: return False

    def _gpu_info(self):
        if not self.has_gpu: return {}
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,name", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
            if r.returncode == 0:
                p = r.stdout.strip().split(", ")
                return {"used": float(p[0]), "total": float(p[1]), "util": float(p[2]), "name": p[3] if len(p) > 3 else "GPU"}
        except: pass
        return {}

    def _cpu_info(self):
        try:
            import psutil; m = psutil.virtual_memory()
            return {"used": m.used/1048576, "total": m.total/1048576, "pct": m.percent, "cpu": psutil.cpu_percent(interval=0.1)}
        except ImportError:
            try:
                r = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=2)
                if r.returncode == 0:
                    p = r.stdout.strip().split("\n")[1].split()
                    return {"used": float(p[2]), "total": float(p[1]), "pct": float(p[2])/float(p[1])*100, "cpu": 0}
            except: pass
        return {}

    def take_snapshot(self):
        if not self._gpu_checked: self.has_gpu = self._detect_gpu(); self._gpu_checked = True
        g, c = {}, {}
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="res") as pool:
            fg, fc = pool.submit(self._gpu_info), pool.submit(self._cpu_info)
            try: g = fg.result(timeout=3)
            except: pass
            try: c = fc.result(timeout=3)
            except: pass
        s = ResourceSnapshot(timestamp=time.time(), gpu_name=g.get("name", ""), gpu_mem_used=g.get("used", 0), gpu_mem_total=g.get("total", 0), gpu_util=g.get("util", 0), cpu_mem_used=c.get("used", 0), cpu_mem_total=c.get("total", 0), cpu_percent=c.get("cpu", 0))
        with self._lock:
            self.snapshot = s; self.history.append(s)
            if len(self.history) > 500: self.history = self.history[-500:]
        return s

    def _loop(self):
        while self._running: self.take_snapshot(); time.sleep(self.interval)

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self): self._running = False

    def get_size(self, model):
        if model in MODEL_SIZES: return MODEL_SIZES[model]
        base = model.split(":")[0]
        for k, v in MODEL_SIZES.items():
            if k.split(":")[0] == base: return v
        return None

    def can_load(self, model):
        sz = self.get_size(model)
        with self._lock: snap = self.snapshot
        r = {"model": model, "can_load": False, "mode": "cpu", "warnings": []}
        if not sz: r["can_load"] = True; return r
        if self.has_gpu and snap.gpu_available >= sz.gpu_mb: r["can_load"], r["mode"] = True, "gpu"
        elif sz.can_cpu and snap.cpu_available >= sz.cpu_mb: r["can_load"], r["mode"] = True, "cpu"
        else: r["warnings"].append(f"Need GPU:{sz.gpu_mb} CPU:{sz.cpu_mb}MB")
        return r

    def suggest_model(self):
        with self._lock: snap = self.snapshot
        for n in ["qwen3:8b","qwen3:4b","qwen2.5:7b","qwen2.5:3b","qwen2.5:1.5b","qwen2.5:0.5b","qwen3:0.6b"]:
            sz = self.get_size(n)
            if sz and ((self.has_gpu and snap.gpu_available >= sz.gpu_mb) or (sz.can_cpu and snap.cpu_available >= sz.cpu_mb)): return n
        return "qwen2.5:0.5b"

    def status(self):
        with self._lock: return self.snapshot.to_dict()
