# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import os
from .perf.efficiency import PerformanceEvalTBG, PerformanceEvalROCm
from .constants import TBG_PERF_GOLD_ROOT, TBG_DATA_ROOT, ROCm_DATA_ROOT, NATIVE_PERF_GOLD_ROOT, ROCM_PERF_GOLD_DATA_ROOT

def initialize_performance_eval_tb():
    perf_evaluator = PerformanceEvalTBG()
    perf_evaluator.ref_folder = NATIVE_PERF_GOLD_ROOT
    print(f"Creating performance evaluation folder at {TBG_PERF_GOLD_ROOT}")
    perf_evaluator(exec_folder=TBG_DATA_ROOT, gen_perf_folder=TBG_PERF_GOLD_ROOT, golden_metrics_folder=NATIVE_PERF_GOLD_ROOT)

def initialize_performance_eval_rocm():
    perf_evaluator = PerformanceEvalROCm()
    perf_evaluator.ref_folder = ROCM_PERF_GOLD_DATA_ROOT
    print(f"Creating performance evaluation folder at {ROCM_PERF_GOLD_DATA_ROOT}")
    perf_evaluator(exec_folder=ROCm_DATA_ROOT, gen_perf_folder=ROCM_PERF_GOLD_DATA_ROOT)

if __name__ == "__main__":
    initialize_performance_eval_tb()
    initialize_performance_eval_rocm()
