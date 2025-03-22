import argparse
import os
import pandas as pd
import torch
from vllm import LLM, SamplingParams    
from vllm.utils import get_open_port
from datasets import load_dataset
from transformers import AutoTokenizer
from multiprocessing import Process
import re
from multiprocessing import Queue
import time
from gsm8k import extract_solution, compute_score


def generate(llm, prompts, use_tqdm=False, args=None):    
    sampling_params = SamplingParams(max_tokens=args.max_tokens,
                                    temperature=args.temperature,
                                    n=args.n)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    responses = [[output.text for output in output_item.outputs] for output_item in outputs]
    return responses


def tp_generate(prompts, args):
    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=args.num_gpus,
        max_model_len=args.max_model_len,
    )
    responses = generate(llm, prompts, use_tqdm=True, args=args)
    return responses

def sub_dp(prompts, DP_size, dp_rank, TP_size, args, results_queue):
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(DP_size)
    os.environ["VLLM_DP_MASTER_IP"] = args.dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(args.dp_master_port)

    # tp_size = 1:
    # dp_rank = 0: 0;
    # dp_rank = 1: 1;
    # tp_size = 2:
    # dp_rank = 0: 0, 1;
    # dp_rank = 1: 2, 3;
    # dp size = # gpus / tp size
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(dp_rank * TP_size, (dp_rank + 1) * TP_size))

    promts_per_rank = len(prompts) // DP_size
    start = dp_rank * promts_per_rank
    end = start + promts_per_rank
    prompts = prompts[start:end]
    if len(prompts) == 0:
        prompts = ["Placeholder"]
    print(f"DP rank {dp_rank} needs to process {len(prompts)} prompts")

    llm = LLM(model=args.model_name, 
              trust_remote_code=True, 
              max_model_len=args.max_model_len,
              tensor_parallel_size=TP_size)
    responses = generate(llm, prompts, use_tqdm=False, args=args)
    print(f"DP rank {dp_rank} finished processing {len(responses)} prompts")
    results_queue.put((dp_rank, start, end, responses))
    print(f'results queue size: {results_queue.qsize()}')
    return responses


def dp_generate(prompts, args):
    DP_size = args.num_gpus
    TP_size = 1

    procs = []
    results_queue = Queue()
    for i in range(DP_size):
        proc = Process(target=sub_dp, args=(prompts, DP_size, i, TP_size, args, results_queue))
        proc.start()
        procs.append(proc)

    all_results = []
    for _ in range(DP_size):
        dp_rank, start, end, responses = results_queue.get()
        all_results.append((dp_rank, start, end, responses))

    for proc in procs:
        proc.join()

    all_results.sort(key=lambda x: x[0])  # 按 dp_rank 排序

    all_responses = []
    for _, start, end, responses in all_results:
        if responses and responses[0][0] != "Placeholder":
            all_responses.extend(responses)
    return all_responses


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
    args.add_argument("--dp_master_ip", type=str, default="127.0.0.1")
    args.add_argument("--dp_master_port", type=int, default=get_open_port())
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
        all_responses = tp_generate(prompts, args)
    elif args.mode == "dp":
        all_responses = dp_generate(prompts, args)
    t1 = time.time()


    total_score = 0
    for example, response in zip(test_parquet, all_responses):
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