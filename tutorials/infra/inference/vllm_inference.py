import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def generate(question_list,model_path):
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(max_tokens=8192,
                                    temperature=0.0,
                                    n=1)
    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)
    completions = [[output.text for output in output_item.outputs] for output_item in outputs]
    return completions
    
def make_conv_hf(question, tokenizer):
    # for math problem
    content = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    # for code problem
    # content = question + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end." 
    msg = [
        {"role": "user", "content": content}
    ]
    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat
    
def run():
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    all_problems = [
        "which number is larger? 9.11 or 9.9?"
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    completions = generate([make_conv_hf(problem_data, tokenizer) for problem_data in all_problems],model_path)
    print(completions)
    
if __name__ == "__main__":
    run()
