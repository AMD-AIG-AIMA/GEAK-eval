# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import argparse
from glob import glob
import numpy as np
import torch
import subprocess
from geak_eval.constants import Names
from loguru import logger

torch.set_printoptions(profile="full")

def compare_pt_files(ref_file_pt_path, gen_file_pt_path, atol=1e-3, rtol=1e-3, verbose=False):  
    """  
    Compare two PyTorch dictionaries saved as .pt files and compute execution accuracy.  
      
    Args:  
        ref_file_pt_path: Reference dictionary path with tensor values  
        gen_file_pt_path: Generated dictionary path with tensor values to compare  
        atol: Absolute tolerance parameter (default: 1e-3)  
        rtol: Relative tolerance parameter (default: 1e-3)  
        verbose: Whether to print comparison details (default: False)  
      
    Returns:  
        tuple: (execution_accuracy_bool, match_stats, error_message)  
               - execution_accuracy_bool: True if all tensors match within tolerance  
               - match_stats: Dictionary with match statistics  
               - error_message: Error message if comparison fails, None otherwise  
    """  
    import torch  
      
    # Check if both inputs are valid  
    if ref_file_pt_path is None:  
        return False, None, "Reference dictionary path doesn't exist"  
    if gen_file_pt_path is None:  
        return False, None, "Generated dictionary path doesn't exist"  
    
    try:
        gen_file_pt = torch.load(gen_file_pt_path, map_location=torch.device('cpu'))  
        ref_file_pt = torch.load(ref_file_pt_path, map_location=torch.device('cpu'))
        del gen_file_pt['_CALL_SUCCESS_'] 
        del ref_file_pt['_CALL_SUCCESS_'] 
    except Exception as e:
        return False, None, f"Error loading PT files: {str(e)}"  
    # Recursive comparison function for nested structures  
    def _compare_tensors(ref, gen, path=""):  
        if isinstance(gen, (list, tuple)):  
            if not isinstance(ref, type(gen)):  
                return False, f"Type mismatch at {path}: ref is {type(ref)}, gen is {type(gen)}"  
            if len(ref) != len(gen):  
                return False, f"Length mismatch at {path}: ref has {len(ref)} items, gen has {len(gen)} items"  
              
            all_match = True  
            for i, (r, g) in enumerate(zip(ref, gen)):  
                item_match, err_msg = _compare_tensors(r, g, f"{path}[{i}]")  
                if not item_match:  
                    all_match = False  
                    if verbose:  
                        print(err_msg)  
                    return False, err_msg  
            return all_match, None  
              
        elif isinstance(gen, dict):  
            if not isinstance(ref, dict):  
                return False, f"Type mismatch at {path}: ref is {type(ref)}, gen is {type(gen)}"  
              
            # Check for key differences  
            ref_keys = set(ref.keys())  
            gen_keys = set(gen.keys())  
              
            if ref_keys != gen_keys:  
                missing = ref_keys - gen_keys  
                extra = gen_keys - ref_keys  
                err_msg = f"Key mismatch at {path}: "  
                if missing:  
                    err_msg += f"Missing keys: {missing} "  
                if extra:  
                    err_msg += f"Extra keys: {extra}"  
                if verbose:  
                    print(err_msg)  
                return False, err_msg  
              
            # Compare each key  
            all_match = True  
            for key in ref_keys:  
                key_path = f"{path}.{key}" if path else key  
                key_match, err_msg = _compare_tensors(ref[key], gen[key], key_path)  
                if not key_match:  
                    all_match = False  
                    if verbose:  
                        print(err_msg)  
                    return False, err_msg  
            return all_match, None  
              
        elif isinstance(gen, torch.Tensor):
            if not isinstance(ref, torch.Tensor):  
                return False, f"Type mismatch at {path}: ref is {type(ref)}, gen is {type(gen)}"  
              
            # Check shape  
            if ref.shape != gen.shape:  
                err_msg = f"Shape mismatch at {path}: ref shape {ref.shape}, gen shape {gen.shape}"  
                if verbose:  
                    print(err_msg)  
                return False, err_msg  
              
            # Check values  
            try:  
                torch.testing.assert_close(ref, gen, atol=atol, rtol=rtol)  
                if verbose:  
                    print(f"PASSED: {path} - Shape: {ref.shape}, dtype: {ref.dtype}")  
                return True, None  
            except Exception as e:  
                err_msg = f"Value mismatch at {path}: {str(e)}"  
                if verbose:  
                    max_diff = torch.max(torch.abs(ref - gen)).item() if ref.numel() > 0 else 0  
                    mean_diff = torch.mean(torch.abs(ref - gen)).item() if ref.numel() > 0 else 0  
                    print(f"FAILED: {path} - {str(e)}")  
                    print(f"  Max abs diff: {max_diff}")  
                    print(f"  Mean abs diff: {mean_diff}")  
                return False, err_msg  
        else:  
            # For non-tensor types  
            if type(ref) != type(gen):  
                return False, f"Type mismatch at {path}: ref is {type(ref)}, gen is {type(gen)}"  
              
            if ref != gen:  
                err_msg = f"Value mismatch at {path}: ref={ref}, gen={gen}"  
                if verbose:  
                    print(err_msg)  
                return False, err_msg  
              
            return True, None  
      
    # Count the number of tensors to compare  
    def count_tensors(obj):  
        if isinstance(obj, torch.Tensor):  
            return 1  
        elif isinstance(obj, dict):  
            return sum(count_tensors(v) for v in obj.values())  
        elif isinstance(obj, (list, tuple)):  
            return sum(count_tensors(item) for item in obj)  
        return 0  
      
    # Collect all tensor comparison results  
    def collect_tensor_results(ref, gen, results=None, path=""):  
        if results is None:  
            results = []  
          
        if isinstance(gen, (list, tuple)):  
            if isinstance(ref, type(gen)) and len(ref) == len(gen):  
                for i, (r, g) in enumerate(zip(ref, gen)):  
                    collect_tensor_results(r, g, results, f"{path}[{i}]")  
        elif isinstance(gen, dict):  
            if isinstance(ref, dict):  
                common_keys = set(ref.keys()).intersection(set(gen.keys()))  
                for key in common_keys:  
                    key_path = f"{path}.{key}" if path else key  
                    collect_tensor_results(ref[key], gen[key], results, key_path)  
        elif isinstance(gen, torch.Tensor) and isinstance(ref, torch.Tensor):  
            try:  
                if ref.shape == gen.shape:  
                    torch.testing.assert_close(ref, gen, atol=atol, rtol=rtol)  
                    results.append((path, True, None))  
                else:  
                    results.append((path, False, f"Shape mismatch: ref {ref.shape}, gen {gen.shape}"))  
            except Exception as e:  
                results.append((path, False, str(e)))  
                  
        return results  
      
    # Main comparison  
    all_match, error_message = _compare_tensors(ref_file_pt, gen_file_pt)  
    # Collect detailed statistics for reporting  
    tensor_results = collect_tensor_results(ref_file_pt, gen_file_pt)  
    total_tensors = len(tensor_results)  
    matched_tensors = sum(1 for _, match, _ in tensor_results if match)  
      
    match_stats = {  
        "total_tensors": total_tensors,  
        "matched_tensors": matched_tensors,  
        "match_percentage": (matched_tensors / total_tensors * 100) if total_tensors > 0 else 0,  
        "detailed_results": tensor_results if verbose else None  
    }  
      
    if verbose:  
        print(f"\nSUMMARY: Total tensors: {total_tensors}, Matched: {matched_tensors}, "  
              f"Match percentage: {match_stats['match_percentage']:.2f}%")  
      
    return all_match, match_stats, error_message  
  
  
def test_pt_correctness(ref_file, gen_file, atol=1e-3, rtol=1e-3, verbose=False):  
    """  
    Test correctness of PyTorch tensors saved in .pt files.  
      
    Args:  
        ref_file: Reference Python file path  
        gen_file: Generated Python file path  
        atol: Absolute tolerance parameter (default: 1e-3)  
        rtol: Relative tolerance parameter (default: 1e-3)  
        verbose: Whether to print comparison details (default: False)  
      
    Returns:  
        tuple: (gen_call_acc, exec_acc, match_stats, gen_stderr)  
               - gen_call_acc: True if generated file was loaded successfully  
               - exec_acc: True if tensors match within tolerance  
               - match_stats: Dictionary with match statistics  
               - gen_stderr: Error message if comparison fails, None otherwise  
    """  
    import torch  
    import os  
      
    fname = os.path.basename(gen_file)  
    gen_call_acc, ref_call_acc = False, False  
    gen_stderr, ref_stderr = None, None  
      
    # Convert file paths to PT paths  
    ref_file_pt_path = ref_file.replace('.', '_') + '.pt'  
    gen_file_pt_path = gen_file.replace('.', '_') + '.pt'  
      
    # Load generated file  
    try:  
        gen_file_pt = torch.load(gen_file_pt_path, map_location=torch.device('cpu'))  
        if verbose:  
            print(f"Successfully loaded generated PT file: {gen_file_pt_path}")     
    except Exception as e:  
        gen_stderr = f"Error loading generated PT file: {str(e)}"  
        return False, False, None, gen_stderr  
      
    # Load reference file  
    try:  
        ref_file_pt = torch.load(ref_file_pt_path, map_location=torch.device('cpu'))  
        if verbose:  
            print(f"Successfully loaded reference PT file: {ref_file_pt_path}")   
    except Exception as e:  
        ref_stderr = f"Error loading reference PT file: {str(e)}"  
        return False, False, None, ref_stderr  

    if gen_file_pt['_CALL_SUCCESS_'].item():
        gen_call_acc = True
    else:
        gen_call_acc = False

    del gen_file_pt['_CALL_SUCCESS_'] 
    del ref_file_pt['_CALL_SUCCESS_'] 
    # Compare PT files  
    exec_acc, match_stats, error_msg = compare_pt_files(ref_file_pt_path, gen_file_pt_path, atol=atol, rtol=rtol, verbose=verbose)  
      
    if not exec_acc:  
        gen_stderr = error_msg or f"Generated tensors do not match reference tensors for file: {fname}"  
        
    
    return gen_call_acc, exec_acc, match_stats, gen_stderr 



def test_correctness_rocm(ref_file, gen_file, var_name='result_gold', atol=1e-3, rtol=1e-3, verbose=False, global_timeout=60*60):
    # per_test_timeout = 60*10 # 5 minutes per test
    logger.info("Global timeout: %ds", global_timeout)
    # ref_result_call = subprocess.run([f'python3 -m pytest --capture=sys -k "not test_performance and not test_save_performance_results" {ref_file}'], capture_output=True, text=True, shell=True)
    # gen_result_call = subprocess.run([f'python3 -m pytest --capture=sys -k "not test_performance and not test_save_performance_results" {gen_file}'], capture_output=True, text=True, shell=True)

    # ref_result_call = subprocess.run([f'python3 -m pytest --timeout={per_test_timeout} --capture=sys -k "not test_performance and not test_save_performance_results" {ref_file}'], capture_output=True, text=True, shell=True, timeout=global_timeout)
    # gen_result_call = subprocess.run([f'python3 -m pytest --timeout={per_test_timeout} --capture=sys -k "not test_performance and not test_save_performance_results" {gen_file}'], capture_output=True, text=True, shell=True, timeout=global_timeout)

    ref_result_call = subprocess.run([f'python3 -m pytest --capture=sys -k "not test_performance and not test_save_performance_results" {ref_file}'], capture_output=True, text=True, shell=True, timeout=global_timeout)
    gen_result_call = subprocess.run([f'python3 -m pytest --capture=sys -k "not test_performance and not test_save_performance_results" {gen_file}'], capture_output=True, text=True, shell=True, timeout=global_timeout)


    gen_call_acc, exec_acc, match_stats, gen_stderr = test_pt_correctness(ref_file, gen_file, atol=atol, rtol=rtol, verbose=verbose)  

    print(f"Logs stderr: {gen_result_call.stdout}")
    print(f"Generated call accuracy: {gen_call_acc}")  
    print(f"Execution accuracy: {exec_acc}")  
    if match_stats:  
        print(f"Match percentage: {match_stats['match_percentage']:.2f}%")  
    if gen_stderr:  
        print(f"Error: {gen_stderr}")  
    return gen_call_acc, exec_acc, match_stats, gen_stderr


    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file", "-pf", type=str, required=True)
    parser.add_argument("--ref_file", "-tf", type=str, required=True)
    parser.add_argument("--var_name", type=str, default="result_gold")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-1)
    parser.add_argument("--global_timeout", type=int, default=60*60)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    gen_call_acc, exec_acc, stdout, gen_stderr = test_correctness_rocm(args.ref_file, args.gen_file, args.var_name, atol=args.atol, rtol=args.rtol, verbose=args.verbose, global_timeout=args.global_timeout)
    print(f"{Names.PYTEST_SEPARATOR}")
    print(f"{gen_call_acc}{Names.RET_SEPERATOR}{exec_acc}{Names.RET_SEPERATOR}{stdout}{Names.RET_SEPERATOR}{gen_stderr}")