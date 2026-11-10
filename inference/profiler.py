"""Resource / speed profiling shared by infer_video.py and eval_cholecseg8k.py.

Measured:
    GPU memory   - torch peak allocated (tensors) and reserved (caching allocator),
                   plus the peak device memory seen by NVML (= nvidia-smi, includes
                   the CUDA context)
    GPU util     - NVML "utilization.gpu" sampled in a background thread
                   (% of time at least one kernel was running on the GPU)
    RAM          - peak RSS of the main process, and peak PSS summed over the main
                   process + its children (DataLoader workers). PSS splits shared
                   pages between processes, so forked workers are not double counted.
    Timing       - model time per batch with CUDA events (network forward + score
                   computation + window merge, i.e. predict()), end-to-end wall time,
                   and a separate batch-1 latency benchmark (real-time scenario).
"""

import os
import resource
import threading
import time

import numpy as np
import psutil
import torch


def _nvml_handle(device):
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception as e:  # noqa: BLE001
        print(f"NVML not available ({e}): no GPU utilisation sampling")
        return None, None
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")
    except Exception:  # noqa: BLE001  fall back to the first visible GPU
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(visible) if visible.isdigit() else 0)
    return pynvml, handle


class ResourceMonitor:
    """Samples GPU utilisation / memory and process-tree RAM every `interval` seconds."""

    def __init__(self, device, interval=0.2):
        self.device = device
        self.interval = interval
        self.proc = psutil.Process()
        self.cuda = device.type == "cuda"
        self.nvml, self.handle = _nvml_handle(device) if self.cuda else (None, None)
        self.gpu_util, self.gpu_mem = [], []
        self.peak_rss_main = 0
        self.peak_pss_tree = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample_ram(self):
        rss_main = self.proc.memory_info().rss
        pss = 0
        for p in [self.proc] + self.proc.children(recursive=True):
            try:
                pss += p.memory_full_info().pss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.peak_rss_main = max(self.peak_rss_main, rss_main)
        self.peak_pss_tree = max(self.peak_pss_tree, pss)

    def _run(self):
        while not self._stop.is_set():
            if self.handle is not None:
                self.gpu_util.append(self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
                self.gpu_mem.append(self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used)
            self._sample_ram()
            self._stop.wait(self.interval)

    def start(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        self._sample_ram()
        gb = 1024 ** 3
        out = {
            "peak_ram_main_rss_gb": self.peak_rss_main / gb,
            "peak_ram_total_pss_gb": self.peak_pss_tree / gb,
            # lifetime peak of the main process, includes loading the checkpoint
            "peak_ram_main_lifetime_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / gb,
        }
        if self.cuda:
            out["peak_gpu_mem_allocated_gb"] = torch.cuda.max_memory_allocated(self.device) / gb
            out["peak_gpu_mem_reserved_gb"] = torch.cuda.max_memory_reserved(self.device) / gb
            out["gpu_total_mem_gb"] = torch.cuda.get_device_properties(self.device).total_memory / gb
        if self.gpu_util:
            util = np.array(self.gpu_util, dtype=float)
            out["peak_gpu_mem_nvml_gb"] = max(self.gpu_mem) / gb
            out["gpu_util_mean_pct"] = float(util.mean())
            out["gpu_util_median_pct"] = float(np.median(util))
            out["gpu_util_max_pct"] = float(util.max())
            out["gpu_util_samples"] = len(util)
        return out


class CudaTimer:
    """Accumulates GPU time of a code region with CUDA events (no sync per batch)."""

    def __init__(self, device):
        self.cuda = device.type == "cuda"
        self.pairs = []
        self.cpu_total = 0.0

    def __enter__(self):
        if self.cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            self._end.record()
            self.pairs.append((self._start, self._end))
        else:
            self.cpu_total += time.perf_counter() - self._t0

    def total_s(self):
        if self.cuda:
            torch.cuda.synchronize()
            return sum(s.elapsed_time(e) for s, e in self.pairs) / 1000.0
        return self.cpu_total


def count_params(network):
    return sum(p.numel() for p in network.parameters())


@torch.no_grad()
def benchmark_latency(predict_fn, network, img_size, resized_hw, device, amp, n_warmup=20, n_iter=200):
    """Latency of ONE frame (batch 1), already resized and on the GPU: pure model time,
    no disk I/O or decoding. This is the number that matters for real-time use."""
    from statistics import median

    img = torch.randint(0, 256, (1, 3, *resized_hw), dtype=torch.uint8, device=device)
    t_idx = torch.zeros(1, dtype=torch.long, device=device)
    times = []
    for i in range(n_warmup + n_iter):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            predict_fn(network, img, img_size, t_idx, None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= n_warmup:
            times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {
        "latency_bs1_mean_ms": float(times.mean()),
        "latency_bs1_median_ms": float(median(times)),
        "latency_bs1_p95_ms": float(np.percentile(times, 95)),
        "fps_bs1": float(1000.0 / times.mean()),
    }


def resolution_info(orig_wh, resized_hw, img_size, num_windows):
    return {
        "input_resolution": f"{orig_wh[0]}x{orig_wh[1]}",
        "model_resized_resolution": f"{resized_hw[1]}x{resized_hw[0]}",
        "window_resolution": f"{img_size[1]}x{img_size[0]}",
        "windows_per_frame": num_windows,
        "output_resolution": f"{orig_wh[0]}x{orig_wh[1]}",
    }


def warmup(predict_fn, network, img_size, resized_hw, device, amp, batch_size, n=3):
    """A few forward passes so that cuDNN/cuBLAS setup is not counted in the timings."""
    img = torch.randint(0, 256, (batch_size, 3, *resized_hw), dtype=torch.uint8, device=device)
    t_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        for _ in range(n):
            predict_fn(network, img, img_size, t_idx, None)
    if device.type == "cuda":
        torch.cuda.synchronize()
