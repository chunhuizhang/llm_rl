
import argparse
import time
import ray
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer
from gsm8k import extract_solution, compute_score

@ray.remote(num_gpus=1)
def generate(prompts, args):

    sampling_params = SamplingParams(max_tokens=args.max_tokens,
                                    temperature=args.temperature,
                                    n=args.n)
    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        # tensor_parallel_size=args.num_gpus,
        max_model_len=args.max_model_len,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    responses = [[output.text for output in output_item.outputs] for output_item in outputs]
    return responses

if __name__ == "__main__":


    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args.add_argument("--num_gpus", type=int, default=2)
    # tp or dp
    args.add_argument("--mode", type=str, default="tp")
    args.add_argument("--data_path", type=str, default="./data/gsm8k_test.parquet")
    args.add_argument("--temperature", type=float, default=0.0)
    args.add_argument("--max_tokens", type=int, default=8192)
    args.add_argument("--max_model_len", type=int, default=4096)
    args.add_argument("--n", type=int, default=1)
    args.add_argument("--extract_method", type=str, default="strict")
    args.add_argument("--num_prompts", type=int, default=-1)
    args = args.parse_args()

    ray.init()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    test_parquet = load_dataset('parquet', data_files=args.data_path)['train']
    prompts = []
    for example in test_parquet:
        prompt = [example['prompt'][0]]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    if args.num_prompts != -1:
        prompts = prompts[:args.num_prompts]
    
    t0 = time.time()
    
    outputs = []
    # Split prompts into batches for each GPU
    batch_size = (len(prompts) + args.num_gpus - 1) // args.num_gpus  # Ceiling division
    for i in range(args.num_gpus):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        if start_idx < len(prompts):
            outputs.append(generate.remote(prompts[start_idx:end_idx], args))

    # list of 2 lists
    outputs = ray.get(outputs)
    outputs = [item for sublist in outputs for item in sublist]
    t1 = time.time()

    total_score = 0
    for example, response in zip(test_parquet, outputs):
        gt_answer = example['reward_model']['ground_truth']
        model_resp = response[0]
        model_answer = extract_solution(model_resp, args.extract_method)
        score = compute_score(model_resp, gt_answer, args.extract_method)
        print(f"Example: {example['prompt'][0]}")
        print(f"Response: {model_resp}")
        print(f"Solution: {model_answer}")
        print(f"Score: {score}")
        print("-"*100)
        total_score += score

    print(f"accuray: {total_score}/{len(prompts)} = {total_score / len(prompts)}")
    print(f"Time taken of {args.mode} mode: {t1 - t0} seconds")