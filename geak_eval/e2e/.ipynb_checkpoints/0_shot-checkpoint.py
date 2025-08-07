#!/usr/bin/env python3
import os
import sys
import json
import argparse
from typing import List, Dict, Any
import requests
from tqdm import tqdm
from pathlib import Path

# Base model interface
class BaseModel:
    def generate(self, messages: List, **kwargs) -> str:
        raise NotImplementedError

# Local OpenAI compatible model
from typing import List
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

class LocalOpenAIModel(BaseModel):
    def __init__(self, 
                 model_id: str = None, 
                 api_key: str = "None", 
                 base_url: str = "http://127.0.0.1:8000/v1"):
        self.api_key = api_key
        self.base_url = base_url

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        if model_id:
            self.model_id = model_id
        else:
            print("model_id not provided, attempting to discover from endpoint...")
            try:
                models = self.client.models.list()
                if not models.data:
                    raise ValueError("No models found at the specified endpoint.")
                self.model_id = models.data[0].id
                print(f"Discovered and set model_id to: {self.model_id}")
            except Exception as e:
                print(f"Error: Could not automatically discover model ID. Please provide it manually.")
                raise e
    
    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List, 
                 temperature: float = 1.0, 
                 presence_penalty=0, 
                 frequency_penalty=0,
                 max_tokens: int = 4096) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            n=1,
            stream=False,
            stop=None,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=None,
            user=None
        )
        if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
            raise ValueError("No response choices returned from the API.")
        return response.choices[0].message.content.strip()

# Claude model
class ClaudeModel(BaseModel):
    def __init__(self, 
                 model_id="claude-3.7", 
                 api_key=None):
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id
        self.SERVER = "https://llm-api.amd.com/claude3"
        self.headers = {
            'Ocp-Apim-Subscription-Key': api_key
        }
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List,
                 temperature=0,
                 presence_penalty=0, 
                 frequency_penalty=0, 
                 max_tokens=50000,
                 max_completion_tokens=50000) -> str:
        body = {
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "max_completion_tokens": max_completion_tokens,
            "max_tokens": max_tokens,
            "presence_Penalty": presence_penalty,
            "frequency_Penalty": frequency_penalty,
        }
        response = requests.post(
            url=f"{self.SERVER}/{self.model_id}/chat/completions",
            json=body,
            headers=self.headers
        )
        return response.json()['content'][0]['text']

# Gemini model
class GeminiModel(BaseModel):
    def __init__(self, 
                 model_id="gemini-2.5-pro", 
                 api_key=None):
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id
        self.SERVER = "https://llm-api.amd.com/vertex/gemini"
        self.HEADERS = {"Ocp-Apim-Subscription-Key": api_key}
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List, 
                 temperature=1.0, 
                 presence_penalty=0, 
                 frequency_penalty=0, 
                 max_tokens=80000) -> str:
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "reasoning_effort": "low",
            "top_P": 0.95,
            "presence_Penalty": presence_penalty,
            "frequency_Penalty": frequency_penalty,
        }
        response_gemini = requests.post(
            url=f"{self.SERVER}/{self.model_id}/chat", 
            json=body,
            headers=self.HEADERS
        )
        code_chat_completion_result = response_gemini.json()
        return code_chat_completion_result['candidates'][0]['content']['parts'][0]['text']

class LLMEvaluationPipeline:
    def __init__(self, model_type: str, model_id: str = None, api_key: str = None, base_url: str = None):
        self.model_type = model_type
        self.model = self._initialize_model(model_type, model_id, api_key, base_url)
    
    def _initialize_model(self, model_type: str, model_id: str, api_key: str, base_url: str) -> BaseModel:
        """Initialize the appropriate model based on model_type"""
        if model_type.lower() == 'local':
            return LocalOpenAIModel(
                model_id=model_id,
                api_key=api_key or "None",
                base_url=base_url or "http://127.0.0.1:8000/v1"
            )
        elif model_type.lower() == 'claude':
            return ClaudeModel(
                model_id=model_id or "claude-3.7",
                api_key=api_key
            )
        elif model_type.lower() == 'gemini':
            return GeminiModel(
                model_id=model_id or "gemini-2.5-pro",
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def load_dataset(self, dataset: str) -> List[Dict[str, Any]]:
        """Load the appropriate dataset"""
        if dataset == 'tbg':
            path = 'data/TritonBench/data/TritonBench_G_comp_alpac_v1_fixed_with_difficulty.json'
        elif dataset == 'rocm':
            path = 'data/ROCm/data/ROCm_eval_complex_instruct_v1.json'
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} samples from {dataset} dataset")
        return data
    
    def generate_response(self, instruction: str) -> str:
        """Generate response from the model"""
        messages = [{"role": "user", "content": instruction}]
        try:
            response = self.model.generate(messages)
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def extract_test_code(self, output_code: str) -> str:
        """Extract test code by splitting on separator and taking second block"""
        separator = '#' * 146
        parts = output_code.split(separator)
        if len(parts) >= 2:
            return parts[1].strip()
        return ""
    
    def process_tbg_dataset(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process TBG dataset and generate responses"""
        results = []
        
        for item in tqdm(data, desc="Processing TBG dataset"):
            instruction = item.get('instruction', '')
            label = item.get('output', '')
            filename = item.get('file', '')
            difficulty = item.get('difficulty', -1)
            
            # Generate response
            predict = self.generate_response(instruction)
            
            # Extract test code
            test_code = self.extract_test_code(label)
            
            result = {
                'instruction': instruction,
                'label': label,
                'filename': filename,
                'difficulty': difficulty,
                'test_code': test_code,
                'predict': predict
            }
            results.append(result)
        
        return results
    
    def process_rocm_dataset(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process ROCm dataset and generate responses"""
        results = []
        
        for item in tqdm(data, desc="Processing ROCm dataset"):
            instruction = item.get('instruction', '')
            label = item.get('label', '')
            file = item.get('file', '')
            target_kernel_name = item.get('target_kernel_name', '')
            
            # Generate response
            predict = self.generate_response(instruction)
            
            # Extract test code
            test_code = self.extract_test_code(label)
            
            result = {
                'instruction': instruction,
                'label': label,
                'file': file,
                'target_kernel_name': target_kernel_name,
                'predict': predict,
                'test_code': test_code
            }
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], dataset: str, output_dir: str = "output"):
        """Save results to JSON file"""
        # Create filename with model name
        model_name = getattr(self.model, 'model_id', self.model_type)
        filename = f"{dataset}_{model_name}_results.json"
        filepath = os.path.join(output_dir, filename)
        # Make sure the full parent directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save the file
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {filepath}")
        return filepath
    
    def run_evaluation(self, results_file: str, dataset: str):
        """Run the evaluation using the existing evaluation code"""
        from geak_eval.run import eval_entry_point
        
        # Create arguments for evaluation
        class Args:
            def __init__(self):
                self.folder_or_file = results_file
                self.outfile = f"eval_results_{dataset}"
                self.dataset = dataset  # 'tbg' or 'rocm'
                self.file_pat = "*"
                self.k_vals = "1,5,10"
                self.run_on_code = False
                self.custom_tests_path = None
                self.debug = 0
        
        args = Args()
        
        try:
            eval_entry_point(args)
            print(f"Evaluation completed for {args.dataset}")
        except Exception as e:
            print(f"Error running evaluation: {e}")

def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Pipeline")
    parser.add_argument('--dataset', '-d', type=str, choices=['tbg', 'rocm'], required=True,
                       help='Dataset to use: tbg or rocm')
    parser.add_argument('--model_type', '-m', type=str, choices=['local', 'claude', 'gemini'], required=True,
                       help='Model type to use')
    parser.add_argument('--model_id', type=str, help='Model ID')
    parser.add_argument('--api_key', type=str, help='API key for the model')
    parser.add_argument('--base_url', type=str, default="http://localhost:8000/v1",help='Base URL for local model')
    parser.add_argument('--output_dir', '-o', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--run_eval', action='store_true',
                       help='Run evaluation after generation')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation (skip generation)')
    parser.add_argument('--results_file', type=str,
                       help='Results file for evaluation (required if --eval_only)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.eval_only and not args.results_file:
        parser.error("--results_file is required when using --eval_only")
    
    if not args.eval_only:
        # Initialize pipeline
        pipeline = LLMEvaluationPipeline(
            model_type=args.model_type,
            model_id=args.model_id,
            api_key=args.api_key,
            base_url=args.base_url
        )
        
        # Load dataset
        data = pipeline.load_dataset(args.dataset)
        # data = data[:1]
        # Process dataset based on type
        if args.dataset == 'tbg':
            results = pipeline.process_tbg_dataset(data)
        else:  # rocm
            results = pipeline.process_rocm_dataset(data)
        
        # Save results
        results_file = pipeline.save_results(results, args.dataset, args.output_dir)
    else:
        results_file = args.results_file
    
    # Run evaluation if requested
    if args.run_eval or args.eval_only:
        if not os.path.exists(results_file):
            print(f"Results file not found: {results_file}")
            return
        
        pipeline_eval = LLMEvaluationPipeline('local')  # Dummy pipeline for evaluation
        pipeline_eval.run_evaluation(results_file, args.dataset)

if __name__ == "__main__":
    main()
