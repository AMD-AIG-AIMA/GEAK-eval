# Technical report [![arXiv](https://img.shields.io/badge/arXiv-2507.23194-b31b1b.svg)](https://arxiv.org/abs/2507.23194)

# Improved TritonBench evaluation framework

### Dependancy installation
- You may install requirements as `pip install -r requirements.txt`

### Installation
Please install running the following command from the root folder:
- `pip install -e .`

### Running evaluation
Before running evaluations you must run the setup to record ground truth performance data for your GPU.
 - `geak-eval setup -ds tbg`
 - `geak-eval setup -ds rocm`

You can run evaluations in the following two ways:
1. Command line run:
    - `geak-eval -f PATH_TO_FOLDER_OR_FILE -o NAME_OF_OUTPUT_FILE -ds tbg` for Tritonbench-G-v1
    - `geak-eval -f PATH_TO_FOLDER_OR_FILE -o NAME_OF_OUTPUT_FILE -ds rocm` for ROCm
2. From python script: the following is a bare minimum example, for a detail example please see `geak-eval/run.py`.
    - `from geak-eval.evaluators.interface import get_evaluators`
    - `evaluator = get_evaluators["tbg"]() # for TritonBenchG eval`
    - `evaluator = get_evaluators["rocm"]() # for ROCm eval`
    - `call_status, exec_status, stdout, stderr = evaluator(generated_code, log_root=PATH_TO_LOG, file_name="kernel.py", atol=1e-5, rtol=1e-2) # run evaluations`

### Issues with existing TritonBench evaluation framework
1. `1_exec_acc.py` file in TritonBench did not accurately compare the outputs of two Triton files.
1. The execution was purely done using subprocess call for both generated and ground truth files.
1. The seed consistancy is violated.
1. The outputs of the two Triton runs are compared using stdout string comparison, which is not always correct.
1. Around ground truth 150 files do not `print(result_gold)` line, hence the eval framework is essentially comapring the two null strings.
1. Some of the ground truth files (e.g. `context_attn_bloom.py`) does not even have `result_gold = test_*()` line at the end. So the call accuracy run using this file `0_call_acc.py` just blindly assumes that the call was success.
1. 7 kernel files (originally provided) run into `memory access faults`, we have fixed them.

### We have fixed these issues as follows:
1. Use `torch.allclose` to compare two runs (ground truth and generated).
1. Fix ground truth files to include `result_gold = test_*()`.
1. Ensure consistent seed across files.


We have also integrated performance measurement into the framework. Kernel evaluation flow is as follows:
1. Check if the kernel is callable: run the test function of the kernel.
2. If the kernel is callable then check if the kernel matches ground truthe by comparing outputs of the generated kernel on know tests.
3. If the generated kernel is correct: run the performance evaluation.

#### Help/support/contribute:
Please raise github issue or PR for any issues/help or contributions!

You can contribute in the following ways:
1. Add new kernels for evaluations: 
    - Add the dataset of new kernels under `geak-eval/data`.
    - Add the path of this new dataset in `geak-eval.constants`.
    - Add an evaluator interface for this new dataset in `geak-eval.evaluators.interface`.
    - Add an evaluator to be run by the interface in `geak-eval.evaluators`. The evaluator is a function that only runs python call and does not run if imported as a module. The `evaluator` (e.g. `TB_correctness.py`) is run by its `interface` (e.g. `interface.TestAllCloseEvaluatorTBG`).  
2. You can add new metrics for evaluator to work with in `geak-eval.metrics`.
3. You can add new performance eval metrics for your (or existing) dataset under `geak-eval.perf`.

### Updates
* [2025-07-16] Added autotune compatible ROCm kernels and naive softmax, use `-tp` argument with path to this folder as below:
    - `geak-eval eval -f PATH_TO_EVAL_FOLDER -o RESULT_NAME -ds rocm -tp geak-eval/data/ROCm/data/ROCm_v1_autotune`
    - `naive_softmax.py` kernel from [rocm blog](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/triton_kernel_dev.html#naive-version) is added to this repo.
    - Use `-c` argument to directly run evaluations on python triton code file(s)/folder instead of json-based parsing.


#### Credits:
Our repo has found the following repos as helpful:
1. [TritonBench](https://github.com/thunlp/TritonBench/tree/main)
2. [ROCm AITER](https://github.com/ROCm/aiter)
3. [ROCm Triton](https://github.com/ROCm/triton)


#### Citation
If you find this work useful in your research or applications, please consider citing:

```bibtex
@misc{wang2025geakintroducingtritonkernel,
      title={Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks}, 
      author={Jianghui Wang and Vinay Joshi and Saptarshi Majumder and Xu Chao and Bin Ding and Ziqiong Liu and Pratik Prabhanjan Brahma and Dong Li and Zicheng Liu and Emad Barsoum},
      year={2025},
      eprint={2507.23194},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.23194}, 
}
```
