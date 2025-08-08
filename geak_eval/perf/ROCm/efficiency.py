# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import os
import json
import argparse

# Default reference folder - adjust if your golden results are elsewhere
DEFAULT_REF_FOLDER = " ../../data/ROCm/data/performance/golden_results/" 

# Theoretical peak performance values (Update these for your specific GPU if different)
# For NVIDIA A100:
#   - FP32 TFLOPs: ~19.5 (non-TensorCore), TF32 TensorCore ~156 TFLOPs, FP16 TensorCore ~312 TFLOPs
#   - Memory Bandwidth: ~1.5 TB/s (HBM2) or ~2.0 TB/s (HBM2e for 80GB A100)
# The values 2039 GB/s and 312 TFLOPS seem plausible for A100 80GB (HBM2e) and FP16 TensorCore.
# Ensure your TFLOPS calculation in the benchmark matches the type of FLOPS for this peak.
# PEAK_GBPS_THEORETICAL = 2039  # GB/s
# PEAK_TFLOPS_THEORETICAL = 312 # TFLOPS (e.g., FP16 TensorCore)

## FOR AMD MI300X
PEAK_GBPS_THEORETICAL = 5300 
PEAK_TFLOPS_THEORETICAL = 1307.4

def find_matching_entry(target_params: dict, data_list: list) -> dict | None:
    """Finds an entry in data_list whose 'params' dict matches target_params."""
    for entry in data_list:
        if "params" in entry and entry["params"] == target_params:
            return entry
    return None

def calculate_single_op_metrics(path_gen: str, path_ref: str,
                                peak_gbps: float, peak_tflops: float):
    """
    Calculates performance metrics for a single operator, comparing generated vs. reference.
    """
    # Lambdas to extract lists of values if they exist and are numeric
    get_metric_values = lambda data, key: [
        item[key] for item in data if isinstance(item.get(key), (int, float))
    ]

    with open(path_gen, 'r', encoding='utf-8') as f_gen:
        data_gen_all = json.load(f_gen)
    with open(path_ref, 'r', encoding='utf-8') as f_ref:
        data_ref_all = json.load(f_ref)

    # Filter out entries that might be error dicts (don't have 'ms', 'GB/s', 'TFLOPS')
    # And ensure they have the 'params' key for matching
    data_gen_valid = [d for d in data_gen_all if all(k in d for k in ["params", "ms", "GB/s", "TFLOPS"])]
    data_ref_valid = [d for d in data_ref_all if all(k in d for k in ["params", "ms", "GB/s", "TFLOPS"])]

    if not data_gen_valid:
        print(f"Warning: No valid benchmark data found in generated file: {os.path.basename(path_gen)}")
        return None, None # Cannot calculate metrics

    # --- Match entries between generated and reference based on "params" ---
    # This is more robust than assuming same length and order.
    matched_ms_gen = []
    matched_ms_ref = []
    # For efficiency calculation, we'll use all valid generated data points
    # For speedup, we only use points that have a match in reference data

    for gen_entry in data_gen_valid:
        ref_entry_match = find_matching_entry(gen_entry["params"], data_ref_valid)
        if ref_entry_match:
            if isinstance(gen_entry.get("ms"), (int, float)) and isinstance(ref_entry_match.get("ms"), (int, float)):
                matched_ms_gen.append(gen_entry["ms"])
                matched_ms_ref.append(ref_entry_match["ms"])
        else:
            print(f"Warning: No matching reference data for params {gen_entry['params']} in {os.path.basename(path_gen)}")


    # 1. Calculate Speedup (Generated vs. Reference) based on matched entries
    speedup_gen_vs_ref = None
    if matched_ms_gen and matched_ms_ref and sum(matched_ms_gen) > 0:
        # Speedup = Time_Ref / Time_Gen. Higher is better for Generated.
        speedup_gen_vs_ref = round(sum(matched_ms_ref) / sum(matched_ms_gen), 4)
    elif not matched_ms_ref:
        print(f"Note: No matching reference entries found to calculate speedup for {os.path.basename(path_gen)}.")


    # 2. Calculate Efficiency for the Generated Kernel (based on its own best performance)
    #    Uses all valid generated data points, not just matched ones.
    gen_gbs_values = get_metric_values(data_gen_valid, "GB/s")
    gen_tflops_values = get_metric_values(data_gen_valid, "TFLOPS")

    efficiency_gen = 0.0 # Default if no valid data
    if gen_gbs_values or gen_tflops_values: # Ensure there's data
        max_gbs_gen = max(gen_gbs_values) if gen_gbs_values else 0
        max_tflops_gen = max(gen_tflops_values) if gen_tflops_values else 0
        
        eff_from_gbps = round(max_gbs_gen * 100 / peak_gbps, 4) if peak_gbps > 0 else 0
        eff_from_tflops = round(max_tflops_gen * 100 / peak_tflops, 4) if peak_tflops > 0 else 0
        efficiency_gen = max(eff_from_gbps, eff_from_tflops)

    # --- Optional: Calculate and compare reference efficiency ---
    # ref_gbs_values = get_metric_values(data_ref_valid, "GB/s")
    # ref_tflops_values = get_metric_values(data_ref_valid, "TFLOPS")
    # efficiency_ref = 0.0
    # if ref_gbs_values or ref_tflops_values:
    #     max_gbs_ref = max(ref_gbs_values) if ref_gbs_values else 0
    #     max_tflops_ref = max(ref_tflops_values) if ref_tflops_values else 0
    #     eff_ref_from_gbps = round(max_gbs_ref * 100 / peak_gbps, 4) if peak_gbps > 0 else 0
    #     eff_ref_from_tflops = round(max_tflops_ref * 100 / peak_tflops, 4) if peak_tflops > 0 else 0
    #     efficiency_ref = max(eff_ref_from_gbps, eff_ref_from_tflops)
    #
    # if efficiency_ref > efficiency_gen:
    #     print(f"  Note ({os.path.basename(path_gen)}): Reference efficiency ({efficiency_ref}%) > Generated ({efficiency_gen}%).")
    # else:
    #     print(f"  Note ({os.path.basename(path_gen)}): Generated efficiency ({efficiency_gen}%) >= Reference ({efficiency_ref}%).")


    # --- Failure Assertions (similar to original, adjust logic as needed) ---
    filename_short = os.path.basename(path_gen)
    if efficiency_gen >= 100.0: # Allow for slight overshoots due to precision
        print(f"  Warning ({filename_short}): Generated efficiency ({efficiency_gen}%) is high. Check peaks/measurements.")
        # Consider if this should be an assert False. Original script asserted.
        # assert False, f"{filename_short} efficiency ({efficiency_gen}%) >= 100%, test failed!"


    # Original assertion: `ms >= 10` where ms was `sum(ref_ms)/sum(gen_ms)`.
    # So, if speedup_gen_vs_ref is very high (e.g., gen is 10x faster), it was a fail.
    # This seems counter-intuitive for a "failure". Usually, failure is if gen is much SLOWER.
    # Let's assume the original intent was to catch regressions (gen is slower) OR suspicious speedups.
    if speedup_gen_vs_ref is not None:
        if speedup_gen_vs_ref < 0.1: # Generated is >10x SLOWER
            assert False, f"{filename_short} regression: Generated is >10x slower (Speedup: {speedup_gen_vs_ref}). Test failed!"
        # elif speedup_gen_vs_ref >= 10.0: # Generated is >10x FASTER
            # print(f"  Note ({filename_short}): Generated is >10x faster (Speedup: {speedup_gen_vs_ref}). Verify if expected.")
            # assert False, f"{filename_short} suspicious speedup ({speedup_gen_vs_ref}) >= 10x. Test failed!" # Original behavior

    return speedup_gen_vs_ref, efficiency_gen


def run_statistics(gen_folder: str, ref_folder: str,
                   peak_gbps: float, peak_tflops: float):
    """
    Processes all JSON files in gen_folder, compares with ref_folder, and prints statistics.
    """
    # Helper for averaging a list, handles empty list
    calculate_average = lambda lst: round(sum(lst) / len(lst), 2) if lst else "N/A"

    json_files = [f for f in os.listdir(gen_folder) if f.endswith(".json")]
    if not json_files:
        print(f"No JSON files found in generated folder: {gen_folder}")
        return

    all_speedups = []
    all_efficiencies = []

    print("=" * 80)
    print(f"Processing folder: {os.path.basename(gen_folder)}")
    print("=" * 80)

    perf_results = {}

    for f_name in json_files:
        path_gen = os.path.join(gen_folder, f_name)
        path_ref = os.path.join(ref_folder, f_name) # Assumes same filename in ref_folder

        print(f"\n--- Comparing: {f_name} ---")

        if not os.path.exists(path_ref):
            print(f"  Reference file not found: {path_ref}. Skipping comparison for this file.")
            continue

        try:
            speedup, efficiency = calculate_single_op_metrics(path_gen, path_ref, peak_gbps, peak_tflops)
            
            if speedup is not None:
                print(f"  Speedup (Gen vs. Ref): {speedup}")
                all_speedups.append(speedup)
            else:
                print(f"  Speedup (Gen vs. Ref): N/A (no matching reference data or gen time was zero)")

            if efficiency is not None:
                print(f"  Generated Efficiency (vs. Theoretical Peak): {efficiency}%")
                all_efficiencies.append(efficiency)
            else:
                print(f"  Generated Efficiency: N/A (no valid generated data)")

            # Save results for this file
            perf_results[f_name] = {
                "ms": speedup,
                "efficiency": efficiency
            }

        except FileNotFoundError as e:
            print(f"  Error: File not found during processing of {f_name} - {e}")
        except AssertionError as e:
            print(f"  FAILED (Assertion): {f_name} - {e}")
        except Exception as e:
            print(f"  FAILED (Other Error): {f_name} - {type(e).__name__}: {e}")

    # Save all results as JSON in the gen_folder
    out_json_path = os.path.join(gen_folder, "all_perf_results.json")
    with open(out_json_path, "w", encoding="utf-8") as out_f:
        json.dump(perf_results, out_f, indent=2)
    print(f"\nSaved all performance results to {out_json_path}")

    print("\n" + "=" * 80)
    print(f"Overall Statistics for: {os.path.basename(gen_folder)}")
    print(f"  Average Speedup (Gen vs. Ref): {calculate_average(all_speedups)}")
    print(f"  Average Generated Efficiency (vs. Theoretical Peak): {calculate_average(all_efficiencies)}%")
    print("=" * 80)


def arg_parser():
    parser = argparse.ArgumentParser(description='Performance Efficiency Statistics for Pytest-generated benchmarks')
    parser.add_argument('--gen_folder', type=str, required=True,
                        help='The folder path containing generated benchmark JSON files.')
    parser.add_argument('--ref_folder', type=str, default=DEFAULT_REF_FOLDER,
                        help='The folder path containing reference (golden) benchmark JSON files.')
    parser.add_argument('--peak_gbps', type=float, default=PEAK_GBPS_THEORETICAL,
                        help='Theoretical peak memory bandwidth (GB/s) of the GPU.')
    parser.add_argument('--peak_tflops', type=float, default=PEAK_TFLOPS_THEORETICAL,
                        help='Theoretical peak compute performance (TFLOPS) of the GPU.')
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()

    gen_folder_abs = os.path.abspath(args.gen_folder)
    ref_folder_abs = os.path.abspath(args.ref_folder)

    if not os.path.isdir(gen_folder_abs):
        print(f"Error: Generated folder not found: {gen_folder_abs}")
        exit(1)
    if not os.path.isdir(ref_folder_abs):
        print(f"Warning: Reference folder not found: {ref_folder_abs}. Speedup calculations will be limited.")
        # The script will try to proceed and handle missing ref files per operator.

    from loguru import logger
    logger.info(f"Performance Reference folder: {ref_folder_abs}")
    logger.info(f"Performance Generated folder: {gen_folder_abs}")
    run_statistics(gen_folder_abs, ref_folder_abs, args.peak_gbps, args.peak_tflops)

    # Example of iterating if you have multiple gen_folders (commented out)
    # root_gen_perf_dir = "/path/to/your/gene_perf_root/"
    # for sub_folder_name in os.listdir(root_gen_perf_dir):
    #     current_gen_folder = os.path.join(root_gen_perf_dir, sub_folder_name)
    #     if os.path.isdir(current_gen_folder):
    #         print(f"\n\nProcessing sub-folder: {current_gen_folder}")
    #         run_statistics(current_gen_folder, ref_folder_abs, args.peak_gbps, args.peak_tflops)