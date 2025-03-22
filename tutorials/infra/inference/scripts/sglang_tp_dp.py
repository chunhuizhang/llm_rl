
import time
import sglang as sgl
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from gsm8k import extract_solution, compute_score
import os
os.environ["NCCL_IGNORE_DISABLED_P2P"] = '1'


def generate(llm, prompts, args=None):    
    sampling_params = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    outputs = llm.generate(prompts, sampling_params)
    responses = [output['text'] for output in outputs]
    return responses


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args.add_argument("--num_gpus", type=int, default=2)
    # tp or dp
    args.add_argument("--mode", type=str, default="tp")
    args.add_argument("--data_path", type=str, default="./data/gsm8k_test.parquet")
    args.add_argument("--temperature", type=float, default=0.0)
    args.add_argument("--max_tokens", type=int, default=2048)
    args.add_argument("--max_model_len", type=int, default=4096)
    args.add_argument("--n", type=int, default=1)
    args.add_argument("--extract_method", type=str, default="strict")
    args.add_argument("--num_prompts", type=int, default=-1)
    args = args.parse_args()

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
    if args.mode == "tp":
        llm = sgl.Engine(model_path=args.model_name,
                        dp_size=1,
                        tp_size=args.num_gpus,
                        mem_fraction_static=0.8,
                        enable_p2p_check=True)
    elif args.mode == "dp":
        llm = sgl.Engine(model_path=args.model_name,
                        dp_size=args.num_gpus,
                        tp_size=1)
    all_responses = generate(llm, prompts, args=args)
    t1 = time.time()

    total_score = 0
    for example, response in zip(test_parquet, all_responses):
        gt_answer = example['reward_model']['ground_truth']
        model_resp = response
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