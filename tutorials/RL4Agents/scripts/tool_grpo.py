from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import re
import json


def get_value(x: float) -> float:
    """
    Get the value of the function at x.

    Args:
        x: The input value.

    Returns:
        The value of the function at x.
    """
    return max(-((x / 5) ** 2) + x + 1, 0.0)


def agent_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        # Regex pattern to find the JSON inside <tool_call>...</tool_call>
        match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)
        if not match:
            rewards.append(-100)
            continue

        # Try to parse the JSON content
        try:
            # Parse the JSON content
            json_data = json.loads(match.group(1))
        except json.JSONDecodeError:
            rewards.append(-80)
            continue

        # Check if the function name is "get_value"
        function_name = json_data.get("name", "")
        if function_name != "get_value":
            rewards.append(-60)
            continue

        # Get the value of "x" argument
        value = json_data.get("arguments", {}).get("x")
        if value is None:
            rewards.append(-40)
            continue

        # Check if the value is a float
        if not isinstance(value, (int, float)):
            rewards.append(-20)
            continue

        rewards.append(get_value(float(value)))
    return rewards


dataset = Dataset.from_list(
    [{"prompt": [{"role": "user", "content": "Call the function get_value with any value."}]}] * 200
)


def main():
    training_args = GRPOConfig(
        output_dir="Qwen2.5-0.5B-GRPO-agent",
        logging_steps=5,
        gradient_accumulation_steps=4,
        max_completion_length=128,
        max_prompt_length=128,
        bf16=True,
        log_completions=True,
    )
    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=agent_reward,
        args=training_args,
        train_dataset=dataset,
        tools=[get_value],
    )

    trainer.train()


if __name__ == "__main__":
    main()