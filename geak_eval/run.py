# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
import json
import argparse
from glob import glob
from tqdm import tqdm
import pandas as pd

from .helpers.time import get_time
# code_call_exec_success_allclose, 
from .evaluators.interface import get_evaluators
# extract_code_from_llm_output, 
from .processors.llm import LLMOutputProcessor
# get_fname_difficulty_from_label,
from .helpers.helper import get_fname_difficulty_from_label 
# passk
from .metrics.passk import PassK
from .perf.efficiency import get_perf_evaluators

from .initializations import initialize_performance_eval_tb, initialize_performance_eval_rocm
from .constants import Names, ROCm_DATA_AUTOTUNE_ROOT

def get_parser():
    parser = argparse.ArgumentParser(prog="geak_eval", description="Check correctness of the code.")
    subparsers = parser.add_subparsers(dest='command')

    ## main and default parser
    main_parser = subparsers.add_parser('eval', help='Run the evaluation on a folder or a file')
    main_parser.add_argument('--folder_or_file', '-f', type=str, required=True, help='Folder to check')
    main_parser.add_argument('--outfile', '-o', type=str, required=True, help='Output file to save results')

    main_parser.add_argument('--dataset', '-ds', type=str, choices=['tbg', 'rocm'], required=True, help='Which dataset to use for eval [TritonBench-G-v1: tbg, ROCm: rocm].')
    main_parser.add_argument('--file_pat', '-p', type=str, default="*", help='Folder to run eval on')
    main_parser.add_argument('--k_vals', '-k', type=str, default=None, help='k values in case of parallel runs, this will be ignored if single file is provided. Comma separated values, e.g. 1,2,3,5,10,15')

    main_parser.add_argument('--run_on_code', '-c', action='store_true', help='Directly run on code files instead of json files. This is useful for running on a generated code files directly.')
    main_parser.add_argument('--custom_tests_path', '-tp', type=str, default=None, help='Path to custom tests to run on the code files. This is useful for running on a generated code files directly.')

    main_parser.add_argument('--debug', '-d', type=int, default=0, help='Folder to check')
    main_parser.set_defaults(func=eval)

    ## setup parser
    setup_parser = subparsers.add_parser('setup', help='Setup the evaluation environment')
    setup_parser.add_argument('--dataset', '-ds', type=str, choices=['all', 'tbg', 'rocm'], default="all", help='Which dataset to run setup for: [TritonBench-G-v1: tbg, ROCm: rocm].')
    setup_parser.set_defaults(func=setup)

    if len(sys.argv) == 1 or sys.argv[1] not in {'eval', 'setup'}:
        sys.argv.insert(1, 'eval')

    ## parse the arguments
    args = parser.parse_args()
    return args

def main():
    args = get_parser()
    return args.func(args)

def setup(args):
    if args.dataset in ['tbg']:
        initialize_performance_eval_tb()
    else:
        # initialize_performance_eval_rocm()
        pass
def eval(args):
    ## instantiate objects
    evaluator = get_evaluators[args.dataset]()
    perf_evaluator = get_perf_evaluators[args.dataset]()
    llm_output_processor = LLMOutputProcessor()
    passk = PassK()
    EXT = ".json" if not args.run_on_code else ".py"
    ## sanity checks
    is_folder = os.path.isdir(args.folder_or_file.strip())
    if is_folder:
        if not args.run_on_code:
            files = glob(os.path.join(args.folder_or_file, f'{args.file_pat}.json'), recursive=True)
            assert len(files) > 0, f"No files found in {args.folder_or_file} with pattern {args.file_pat}.json"
        else:
            files = glob(os.path.join(args.folder_or_file, f'{args.file_pat}.py'), recursive=True)
            assert len(files) > 0, f"No files found in {args.folder_or_file} with pattern {args.file_pat}.py"
    else:
        files = [args.folder_or_file.strip()]

    data_across_passes = []
    total_passes = len(files)
    pass_num = -1
    for file in tqdm(files, desc="Processing folder", unit="file"):
        print(f"Processing file: {file}")
        if args.debug > 0:
            if pass_num > 2:
                break
        log_root = os.path.abspath(os.path.join(file.replace(EXT, ""), 'tmp'))
        os.makedirs(log_root, exist_ok=True)

        exec_root = os.path.abspath(os.path.join(file.replace(EXT, ""), 'exec'))
        os.makedirs(exec_root, exist_ok=True)

        out_file = os.path.join(file.replace(EXT, ""), args.outfile)
        logs = []
        call_acc, exec_acc = 0, 0
        eval_data_for_file = []
        pass_num += 1
        with open(file, 'r') as f:
            data = json.load(f) if not args.run_on_code else range(1)
            num_files = 0
            for item in tqdm(data, desc="Processing file", unit="item"):
                if args.debug > 0:
                    if num_files >4:
                        break
                    num_files += 1
                if not args.run_on_code:
                    response = item[Names.PREDICT]
                    code = llm_output_processor(response)
                    if "file" in item:
                        fname = item[Names.FILE]
                        difficulty = item.get(Names.DIFFICULTY, -1)
                    else:
                        fname, difficulty = get_fname_difficulty_from_label(item[Names.LABEL])
                    
                    assert fname is not None, f"File name is None for {item[Names.LABEL]}"
                    assert difficulty is not None, f"Difficulty is None for {item[Names.LABEL]}"
                    # assert code is not None, f"Code is None for {item['label']}" 
                    ## FIXED: Actually if the code is None just for a few prompts, then other prompts should be evaluated
                else:
                    code = open(file, 'r').read().strip()
                    fname = os.path.basename(file)
                    difficulty = -1  # No difficulty for code files
                if code is None:
                    call_status, exec_status, stdout, stderr = False, False, "", "Code is empty"
                else:
                    call_status, exec_status, stdout, stderr = evaluator(code, log_root, exec_root, fname, atol=1e-2, rtol=1e-1, custom_tests_path=args.custom_tests_path)

                eval_data = {
                    Names.PASS_NUM : pass_num,
                    Names.FILE_NAME : fname,
                    Names.CALL_STATUS : 1 if call_status else 0,
                    Names.EXEC_STATUS : 1 if exec_status else 0,
                    Names.STDOUT : stdout,
                    Names.STDERR : stderr,
                    Names.DIFFICULTY : int(difficulty)
                }
                eval_data_for_file.append(eval_data)
                call_acc += 1 if call_status else 0 
                exec_acc += 1 if exec_status else 0 
                log = f"{get_time()} => File: {fname}, Call Status: {call_status}, Exec Status: {exec_status}, difficulty: {difficulty}, stderr: {stderr}"
                logs.append(log)
                print(log.split("stderr")[0])
                with open(out_file, 'w') as out_f:
                    for _log in logs:
                        out_f.write(_log + '\n')
            call_acc /= len(data)
            exec_acc /= len(data)
            with open(out_file, 'a') as out_f:
                _log = f"{get_time()} => File: {file}, Call Accuracy: {call_acc}, Exec Accuracy: {exec_acc}"
                out_f.write(_log + '\n')
        
        perf_data = None
        ## Do the performance evaluation
        try:
            perf_data = perf_evaluator(exec_root) ## returns (speedup, GPU efficiency) for tbg
        except Exception as e:
            print(f"Error: {e}")

        data_across_passes += eval_data_for_file
        # Save the data for this pass to a file
        with open(out_file + f"_results_{pass_num}.json", 'w') as out_f:
            json.dump(eval_data_for_file, out_f, indent=4)

        with open(out_file + f"_perf_{pass_num}.json", 'w') as out_f:
            json.dump(perf_data, out_f, indent=4)

    froot = os.path.join(args.folder_or_file.replace(EXT, ""), args.outfile)
    # Save the data across passes to a file
    with open(froot +  "_all_passes.json", 'w') as out_f:
        json.dump(data_across_passes, out_f, indent=4)

    # Save the data across passes to a CSV file
    df = pd.DataFrame(data_across_passes)

    df = df.groupby(Names.FILE_NAME).agg({Names.CALL_STATUS: 'sum', Names.EXEC_STATUS: 'sum', Names.DIFFICULTY: 'first'}).reset_index()
    df[Names.CALL_STATUS] = df[Names.CALL_STATUS]
    df[Names.EXEC_STATUS] = df[Names.EXEC_STATUS]

    ## now return a dictionary with file_name as key and call_status and exec_status as values
    df = df.set_index(Names.FILE_NAME).T.to_dict()
    df = {k: {Names.CALL_STATUS: v[Names.CALL_STATUS], Names.EXEC_STATUS: v[Names.EXEC_STATUS], Names.DIFFICULTY: v[Names.DIFFICULTY]} for k, v in df.items()}

    call_acc = 0
    exec_acc = 0

    if (total_passes > 1) and (args.k_vals is not None):
        for k_val in [int(_k) for _k in args.k_vals.split(",")]:
            if k_val >= total_passes:
                print(f"Skipping k={k_val} as it is greater than total passes {total_passes}")
                continue
            for k, v in df.items():
                _call_pass = passk(total_passes, int(v[Names.CALL_STATUS]), k_val)
                _exec_pass = passk(total_passes, int(v[Names.EXEC_STATUS]), k_val)
                call_acc += _call_pass
                exec_acc += _exec_pass
            call_acc /= len(df)
            exec_acc /= len(df)
            print(f"Call Accuracy for pass@{k_val}: {100* call_acc}")
            print(f"Exec Accuracy for pass@{k_val}: {100* exec_acc}")
            with open(froot + "passk.txt", 'a') as out_f:
                out_f.write(f"Call Accuracy for k={k_val}: {100 * call_acc}\n")
                out_f.write(f"Exec Accuracy for k={k_val}: {100 * exec_acc}\n")
    elif total_passes == 1:
        for k, v in df.items():
            call_acc += int(v[Names.CALL_STATUS])
            exec_acc += int(v[Names.EXEC_STATUS])
        call_acc /= len(df)
        exec_acc /= len(df)
        print(f"Call Accuracy: {100* call_acc}")
        print(f"Exec Accuracy: {100* exec_acc}")
        with open(froot + "passk.txt", 'a') as out_f:
            out_f.write(f"Call Accuracy: {100 * call_acc}\n")
            out_f.write(f"Exec Accuracy: {100 * exec_acc}\n")
    else:
        print("No k values provided, skipping pass@k evaluation.")
        ## save df instead
        with open(froot + "_summary.json", 'w') as out_f:
            json.dump(df, out_f, indent=4)

if __name__ == "__main__":
    main()
