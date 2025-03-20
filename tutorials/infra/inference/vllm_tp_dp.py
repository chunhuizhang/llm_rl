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


def generate(llm, prompts, use_tqdm=False, args=None):    
    sampling_params = SamplingParams(max_tokens=args.max_tokens,
                                    temperature=args.temperature,
                                    n=args.n)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    responses = [[output.text for output in output_item.outputs] for output_item in outputs]
    return responses

def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']

    if method == 'strict':
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split('#### ')[1].replace(',', '').replace('$', '')
    elif method == 'flexible':
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ['', '.']
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def tp_generate(prompts, args):
    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=args.num_gpus,
        gpu_memory_utilization=0.90,
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
              tensor_parallel_size=TP_size, 
              gpu_memory_utilization=0.8)
    responses = generate(llm, prompts, use_tqdm=False, args=args)
    print(f"DP rank {dp_rank} finished processing {len(responses)} prompts")
    results_queue.put((dp_rank, start, end, responses))
    print(f'results queue size: {results_queue.qsize()}')
    return responses


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args.add_argument("--num_gpus", type=int, default=2)
    # tp or dp
    args.add_argument("--mode", type=str, default="tp")
    args.add_argument("--data_path", type=str, default="./data/gsm8k_test.parquet")
    args.add_argument("--temperature", type=float, default=0.0)
    args.add_argument("--max_tokens", type=int, default=8192)
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

    if args.mode == "tp":
        all_responses = tp_generate(prompts, args)
    
    elif args.mode == "dp":

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