"""Score a checkpoint on CPU and report peak process RAM (Windows working set).

python measure_eval.py runs/final/checkpoint.pt --split test
"""

import argparse
import ctypes
import json
import platform
import time
from pathlib import Path

import torch
from common import load_data, make_model, setup
from evaluate import score


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


def peak_working_set_gb():
    if platform.system() != 'Windows':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    get_memory_info.restype = ctypes.c_bool
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    ok = get_memory_info(get_current_process(), ctypes.byref(counters), counters.cb)
    if not ok:
        return float('nan')
    return counters.PeakWorkingSetSize / 2 ** 30


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--split', default='test')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    device, _ = setup('cpu', 'fp32', args.threads)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(state['implementation'], state['config'], device)
    model.load_state_dict(state['model'])
    data = load_data()
    started = time.perf_counter()
    result = score(model, *data[args.split], device, 'fp32')
    result.pop('window_nll_nats')
    print(json.dumps({'checkpoint': str(args.checkpoint), 'split': args.split,
                      'wall_seconds': round(time.perf_counter() - started, 2),
                      'scoring_seconds': round(result['seconds'], 2),
                      'bpb': result['bpb'], 'token_ppl': result['token_ppl'],
                      'peak_ram_gb': round(peak_working_set_gb(), 3)}, indent=2))


if __name__ == '__main__':
    main()
