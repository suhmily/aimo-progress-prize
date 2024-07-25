# %%
import torch
import gc
from vllm import LLM, SamplingParams
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import re
from collections import Counter
import numpy as np
import time
import concurrent.futures
import os
import subprocess
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
N = 3  # Number of candidates
M = 2  # Depth of generation
BATCH_SIZE = 4  # Batch size for vLLM
NUM_WORKERS = 8  # Number of worker processes
TIME_LIMIT = 18 * 3600  # 18 hours limit
TEST_SIZE = 1  # Number of GSM8K problems to test

# Initialize vLLM
MODEL_PATH = "/nlp_group/decapoda-research/Llama-2-7b-chat-hf"
llm = LLM(model=MODEL_PATH, tensor_parallel_size=1)  # Using all 8 GPUs
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=1024,
    stop=["```", "```output", "```python", "```\nOutput", ")\n```"]
)

def generate_prompt(problem):
    return f"""Solve this math problem using Python:

{problem}

Provide your solution as a Python function named 'solve_problem' that takes no arguments and returns the answer. Store the final answer in a variable named 'answer'. Enclose your code in triple backticks with 'python' specified, like this:

```python
def solve_problem():
    # Your code here
    answer = # Final calculated answer
    return answer
```

Do not include any explanations or additional text outside the code block.
"""

def generate_batch(prompts):
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

def extract_code(completion):
    code_block = re.search(r'```python\s*(.*?)\s*```', completion, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    return None

def execute_code(code):
    try:
        # Add a print statement to ensure the final answer is output
        code += "\nprint(f'The final answer is {solve_problem()}')"
        
        # Create a dictionary to store local variables
        local_vars = {}
        
        # Execute the code in a restricted environment
        exec(code, {'__builtins__': {'print': print, 'int': int, 'float': float}}, local_vars)
        
        # Extract the answer from the local variables
        if 'answer' in local_vars:
            return local_vars['answer']
        else:
            return None
    except Exception as e:
        logging.error(f"Error executing code: {e}")
        return None

def process_candidate(problem):
    prompt = generate_prompt(problem)

    for i in range(M):
        logging.info(f"Attempt {i+1} for problem: {problem}")
        completion = generate_batch([prompt])[0]
        logging.info(f"Model completion:\n{completion}")
        
        code = extract_code(completion)
        if code:
            logging.info(f"Extracted code:\n{code}")
            
            result = execute_code(code)
            if result is not None:
                logging.info(f"Execution result: {result}")
                return result
            
        prompt += f"\n\nYour previous attempt was incorrect or incomplete. Please try again."
    
    logging.warning("Failed to get a valid answer after all attempts")
    return None

def process_problem(item):
    question = item['question']
    true_answer = int(item['answer'].split()[-1])
    logging.info(f"Processing problem: {question}")
    logging.info(f"True answer: {true_answer}")
    
    candidates = [process_candidate(question) for _ in range(N)]
    candidates = [c for c in candidates if c is not None]
    
    logging.info(f"Candidates: {candidates}")
    
    if candidates:
        predicted_answer = Counter(candidates).most_common(1)[0][0]
        logging.info(f"Predicted answer: {predicted_answer}")
        return predicted_answer == true_answer
    
    logging.warning("No valid candidates found")
    return False

def evaluate_gsm8k():
    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"].select(range(TEST_SIZE))  # Only use TEST_SIZE samples
    correct = 0
    total = 0
    start_time = time.time()

    for item in test_data:
        if time.time() - start_time > TIME_LIMIT:
            logging.info("Time limit reached. Stopping evaluation.")
            break
        
        result = process_problem(item)
        if result:
            correct += 1
        total += 1
        
        logging.info(f"Accuracy so far: {correct}/{total} = {correct/total:.2%}")

    final_accuracy = correct / total
    logging.info(f"Final Accuracy: {correct}/{total} = {final_accuracy:.2%}")
    return final_accuracy

# Main execution
if __name__ == "__main__":
    try:
        accuracy = evaluate_gsm8k()
        print(f"Evaluation completed. Final accuracy: {accuracy:.2%}")
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")


