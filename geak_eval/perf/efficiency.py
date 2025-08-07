# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import os
import shutil
import json
from .base import BasePerfEval
from ..helpers.helper import run_shell
from ..constants import TBG_PERF_GOLD_DATA_ROOT, ROCM_PERF_GOLD_DATA_ROOT

class PerformanceEvalTBG(BasePerfEval):
    """
    Performance evaluation for the TBG model.
    This class inherits from BasePerfEval and implements the evaluate method.
    """
    ref_folder = TBG_PERF_GOLD_DATA_ROOT
    def __init__(self, name :str='PerformanceEvalTBG'):
        """
        Initialize the PerformanceEvalTBG instance.
        
        Args:
            name (str): The name of the performance evaluation instance.
        """
        super().__init__(name=name)

    def evaluate(self, exec_folder: str, gen_perf_folder: str=None, golden_metrics_folder:str=None) -> dict:
        """
        Evaluate the performance of the TBG model on the given data.
        
        Args:
            folder: Root location with kernels to evaluate.
        
        Returns:
            A dictionary containing the evaluation results.
        """
        
        ref_folder = self.ref_folder        
        print(f"Running performance analysis for {exec_folder}")
        assert os.path.exists(exec_folder), f"Execution folder {exec_folder} does not exist."

        gen_perf_folder = os.path.join(exec_folder, 'gen_perf') if gen_perf_folder is None else gen_perf_folder
        ## if gen_perf_folder exists, remove it
        if os.path.exists(gen_perf_folder):
            print(f"Removing existing performance folder: {gen_perf_folder}")
            shutil.rmtree(gen_perf_folder)
        os.makedirs( gen_perf_folder, exist_ok=True)

        exec_folder = os.path.abspath(exec_folder)
        gen_perf_folder = os.path.abspath(gen_perf_folder)

        curr_dir = os.path.dirname(os.path.abspath(__file__))

        print("Writing files to the performance folder...")
        cmd = [f'python3 {curr_dir}/run_bench/write_file.py --input_folder_path {exec_folder} --result_folder_path {gen_perf_folder}']
        if golden_metrics_folder:
            cmd[-1] += f' --golden_metrics_folder {golden_metrics_folder}'

        write_status, write_stdout, write_stderr = run_shell(cmd)
        print(f"Write status: {write_status}, stdout: {write_stdout}, stderr: {write_stderr}")
        perf_stdout = None

        if write_status:
            print("Files written successfully to the performance folder. Running them...")
            cmd = [f"python3 {curr_dir}/run_bench/multiprocess_gpu_run.py --root_dir {gen_perf_folder}"]
            mp_run_status, mp_run_stdout, mp_run_stderr = run_shell(cmd)
            print(f"Multiprocess GPU run status: {mp_run_status}, stdout: {mp_run_stdout}, stderr: {mp_run_stderr}")
            if mp_run_status:
                print("Multiprocess GPU run completed successfully. Running performance analysis...")
                cmd = [f"python3 {curr_dir}/2_efficiency.py --gen_folder {gen_perf_folder} --ref_folder {ref_folder}"]
                perf_status, perf_stdout, perf_stderr = run_shell(cmd)
                print(f"Performance analysis status: {perf_status}, stdout: {perf_stdout}, stderr: {perf_stderr}")
                
                if perf_status:
                    print(f"Performance analysis completed successfully for {exec_folder}.")
                    with open(os.path.join(exec_folder, 'performance_analysis.txt'), 'w') as f:
                        f.write(f"Performance analysis for {exec_folder}:\n")
                        f.write(perf_stdout)
                else:
                    assert False, f"Failed to run 2_efficiency.py: {perf_stderr}"
            else:
                assert False, f"Failed to run multiprocess_gpu_run.py: {mp_run_stderr}"

        else:
            assert False, f"Failed to write files: {write_stderr}"
        print(f"DONE with performance analysis for {exec_folder}")

        parser_perf_data = self.parse(gen_perf_folder)

        return parser_perf_data

    def parse(self, perf_data_path:str) -> dict:
        eff_fname = os.path.join(perf_data_path, 'efficiency.json')
        parsed_perf_data = {}
        if os.path.exists(eff_fname):
            with open(eff_fname, 'r') as f:
                perf_data = json.load(f)
            parsed_perf_data = perf_data
        return parsed_perf_data
    
class PerformanceEvalROCm(BasePerfEval):
    """
    Performance evaluation for the ROCm kernels.
    This class inherits from BaseEval and implements the evaluate method.
    """
    ref_folder = ROCM_PERF_GOLD_DATA_ROOT

    def __init__(self, name: str = 'PerformanceEvalROCm', ref_folder: str = None):
        """
        Initialize the PerformanceEvalROCm instance.
        
        Args:
            name (str): The name of the performance evaluation instance.
        """
        super().__init__(name=name)
        if ref_folder is not None:
            self.ref_folder = ref_folder
        else:
            self.ref_folder = ROCM_PERF_GOLD_DATA_ROOT


    def evaluate(self, exec_folder: str, gen_perf_folder: str = None) -> dict:
        """
        Evaluate the performance of ROCm kernels.
        
        Args:
            exec_folder: Root location with kernels (py files with pytest functions) to evaluate.
            gen_perf_folder: Folder where performance JSONs will be stored. Defaults to exec_folder/perf.
        
        Returns:
            A dictionary containing the evaluation results (ms and efficiency for each file).
        """
        
        print(f"Running ROCm performance analysis for {exec_folder}")
        assert os.path.exists(exec_folder), f"Execution folder {exec_folder} does not exist."

        if gen_perf_folder is None:
            gen_perf_folder = os.path.join(exec_folder, 'perf')

        
        if os.path.exists(gen_perf_folder):
            print(f"Removing existing performance folder: {gen_perf_folder}")
            shutil.rmtree(gen_perf_folder)
        os.makedirs(gen_perf_folder, exist_ok=True)
        print(f"Ensured gen_perf_folder exists at: {gen_perf_folder}")

        exec_folder_abs = os.path.abspath(exec_folder)
        gen_perf_folder_abs = os.path.abspath(gen_perf_folder)

        curr_dir = os.path.dirname(os.path.abspath(__file__))

        py_files = [f for f in os.listdir(exec_folder_abs) if os.path.isfile(os.path.join(exec_folder_abs, f)) and f.endswith('.py')]

        if not py_files:
            print(f"No .py files found in {exec_folder_abs}. Skipping pytest execution.")
        else:
            print(f"Found .py files to test: {py_files}")

        for py_file_name in py_files:
            py_file_path = os.path.join(exec_folder_abs, py_file_name)
            
            # Define the specific tests to run in order: test_performance then test_save_performance_results
            tests_to_run_specifiers = f"{py_file_path}::test_performance {py_file_path}::test_save_performance_results"
            
            print(f"Running pytest for {py_file_path} (test_performance then test_save_performance_results)...")
            cmd_pytest = [f"pytest {tests_to_run_specifiers}"]
            
            # Pass gen_perf_folder_abs as an environment variable so tests can use it
            test_env = os.environ.copy()
            test_env["PERF_OUTPUT_DIR"] = gen_perf_folder_abs
            
            global_timeout = 30*60
            
            try:
                pytest_status, pytest_stdout, pytest_stderr = run_shell(cmd_pytest, env=test_env, timeout=global_timeout)
            except TimeoutError as e:
                pytest_status = False
                pytest_stdout = None
                pytest_stderr = str(e)
                print(f"Pytest execution timed out for {py_file_path} (test_performance & test_save_performance_results): {pytest_stderr}") 
                
            print(f"Pytest run for {py_file_path} (test_performance & test_save_performance_results) status: {pytest_status}")
            if pytest_stdout:
                print(f"Stdout: {pytest_stdout}")
            if pytest_stderr:
                print(f"Stderr: {pytest_stderr}")

            if not pytest_status:
                assert False, f"Pytest execution failed for {py_file_path} (test_performance & test_save_performance_results): {pytest_stderr}"
        

        print("All pytest runs completed.")

        efficiency_script_path = os.path.join(curr_dir, "ROCm", "efficiency.py") 
        # Attempt to find ROCm folder if it's a sibling to the 'perf' directory (where this script might be)
        if not os.path.exists(efficiency_script_path):
            # curr_dir is .../perf, so os.path.dirname(curr_dir) is the parent of 'perf'
            alt_efficiency_script_path = os.path.join(os.path.dirname(curr_dir), "ROCm", "efficiency.py")
            if os.path.exists(alt_efficiency_script_path):
                efficiency_script_path = alt_efficiency_script_path
            else:
                 # Check if ROCm is a direct subdirectory of the current script's location
                potential_path = os.path.join(curr_dir, "ROCm", "efficiency.py")
                if os.path.exists(potential_path):
                    efficiency_script_path = potential_path
                else:
                    assert False, f"Efficiency script not found. Checked: {os.path.join(curr_dir, 'ROCm', 'efficiency.py')} and {alt_efficiency_script_path}"
        
        if gen_perf_folder_abs == self.ref_folder:
            # Automatically bootstrap, perf baselines based on GPU
            src_dir = os.path.join(exec_folder_abs, 'perf')
            dst_dir = gen_perf_folder_abs
            os.makedirs(dst_dir, exist_ok=True)
            
            for root, _, files in os.walk(src_dir):
                for file in files:
                    shutil.copy2(os.path.join(root, file), os.path.join(dst_dir, file))
            
            shutil.rmtree(src_dir)
            # dont do further, return
            return {}

        cmd_efficiency = [f"python3 {efficiency_script_path} --gen_folder {gen_perf_folder_abs} --ref_folder {self.ref_folder}"]

        print(f"Running efficiency script: {' '.join(cmd_efficiency)}")
        eff_status, eff_stdout, eff_stderr = run_shell(cmd_efficiency)
        
        print(f"Efficiency script status: {eff_status}")
        if eff_stdout:
            print(f"Stdout: {eff_stdout}")
        if eff_stderr:
             print(f"Stderr: {eff_stderr}")

        if not eff_status:
            assert False, f"Failed to run efficiency script {efficiency_script_path}: {eff_stderr}"
        
        print(f"ROCm performance analysis script completed for {exec_folder}.")
        
        try:
            with open(os.path.join(exec_folder_abs, 'rocm_performance_analysis.txt'), 'w') as f:
                f.write(f"ROCm Performance analysis for {exec_folder_abs}:\n")
                f.write(f"Pytest phase produced outputs in {gen_perf_folder_abs}\n\n")
                f.write("Efficiency script stdout:\n")
                f.write(eff_stdout or "N/A")
                if eff_stderr:
                    f.write("\n\nEfficiency script stderr:\n")
                    f.write(eff_stderr)
        except Exception as e:
            print(f"Failed to write rocm_performance_analysis.txt: {e}")

        parsed_perf_data = self.parse(gen_perf_folder_abs)

        return parsed_perf_data

    def parse(self, perf_data_path: str) -> dict:
        """
        Parse the all_perf_results.json file generated by the ROCm efficiency script.
        
        Args:
            perf_data_path: Path to the folder containing all_perf_results.json (this is gen_perf_folder).
        
        Returns:
            A dictionary containing the parsed performance data (ms and efficiency).
        """
        eff_fname = os.path.join(perf_data_path, 'all_perf_results.json')
        parsed_perf_data = {}
        if os.path.exists(eff_fname):
            print(f"Parsing performance data from {eff_fname}")
            try:
                with open(eff_fname, 'r') as f:
                    perf_data = json.load(f)
                parsed_perf_data = perf_data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {eff_fname}: {e}. Returning empty data.")
            except Exception as e:
                print(f"An unexpected error occurred while parsing {eff_fname}: {e}. Returning empty data.")
        else:
            print(f"Performance results file not found: {eff_fname}. Returning empty data.")
        
        return parsed_perf_data



get_perf_evaluators = {
    'tbg': PerformanceEvalTBG,
    'rocm': PerformanceEvalROCm
}