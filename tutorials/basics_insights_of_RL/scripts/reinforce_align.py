import torch
from torch.distributions import Categorical
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
import os
import matplotlib.pyplot as plt

# Set tokenizer parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

NUM_EPOCHS = 80
# batch_size * gradient_accumulation_steps = 1024
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 32
NUM_TOKENS = 10
# LR = 1e-5
LR = 2e-6
KL_FACTOR = 0
# KL_FACTOR = 6000

# Set memory efficient settings
torch.cuda.empty_cache()  # Clear cache before starting
torch.backends.cudnn.benchmark = True


embedding_model = SentenceTransformer("all-MiniLM-L6-v2").to("cuda")
reference_embedding = embedding_model.encode("cat", convert_to_tensor=True)

for param in embedding_model.parameters():
    param.requires_grad = False


def compute_rewards(sequences):
    sequence_embeddings = embedding_model.encode(sequences, convert_to_tensor=True)
    cosine_similarities = util.pytorch_cos_sim(
        reference_embedding.unsqueeze(0), sequence_embeddings
    ).squeeze()
    return cosine_similarities


model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M").to("cuda")
ref_model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M").to(
    "cuda"
)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
optimizer = AdamW(model.parameters(), lr=LR)

for param in ref_model.parameters():
    param.requires_grad = False

prompt = "Once upon a time there was"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

# Add these lists to store metrics
kl_div_history = []
reward_history = []
steps_history = []
step_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    total_rewards = 0
    
    for accum_step in range(GRADIENT_ACCUMULATION_STEPS):
        output_ids = torch.full(
            (BATCH_SIZE, NUM_TOKENS), tokenizer.eos_token_id, device="cuda"
        )
        output_ids[:, : input_ids.shape[1]] = input_ids

        log_probs_accumulated = torch.zeros((BATCH_SIZE, 1), device="cuda")
        kl_div_accumulated = torch.zeros((BATCH_SIZE, 1), device="cuda")

        active_mask = torch.ones(BATCH_SIZE, dtype=torch.bool, device="cuda")

        for i in range(input_ids.shape[1], NUM_TOKENS):
            prompt = output_ids[:, :i].clone()
            logits = model(prompt).logits[:, -1, :]
            # Only consider logits of active sequences
            logits_active = logits[active_mask]
            if logits_active.shape[0] == 0:
                # All sequences are finished
                break
            probs = torch.nn.functional.softmax(logits_active, dim=-1)
            dist = Categorical(probs)
            next_tokens = dist.sample()
            log_probs_accumulated[active_mask] += dist.log_prob(next_tokens).unsqueeze(-1)
            output_ids[active_mask, i] = next_tokens

            # Compute reference model
            ref_logits = ref_model(prompt).logits[:, -1, :]
            ref_logits_active = ref_logits[active_mask]

            # Compute KL Divergence
            kl_div = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(logits_active, dim=-1),
                torch.nn.functional.log_softmax(ref_logits_active, dim=-1),
                reduction="none",
                log_target=True,
            )
            kl_div_accumulated[active_mask] += kl_div.mean(dim=-1).unsqueeze(-1)

            finished = next_tokens == tokenizer.eos_token_id
            active_indices = torch.nonzero(active_mask).squeeze(-1)
            new_mask = active_mask.clone()
            new_mask[active_indices] = ~finished
            active_mask = new_mask

        normalized_log_probs = log_probs_accumulated / NUM_TOKENS
        normalized_kl_div = kl_div_accumulated / NUM_TOKENS

        # Compute rewards for the entire batch
        with torch.no_grad():
            sequences = [
                tokenizer.decode(input_id, skip_special_tokens=True)
                for input_id in output_ids
            ]
            rewards = compute_rewards(sequences)

        # Compute loss for the entire batch
        neg_advantage = (-normalized_log_probs * rewards.unsqueeze(-1)).mean()
        loss = neg_advantage + KL_FACTOR * normalized_kl_div.mean()

        # Scale loss by gradient accumulation steps
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        loss.backward()
        
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        total_rewards += rewards.mean().item()
        
        if (accum_step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            # Record metrics
            step_counter += 1
            steps_history.append(step_counter)
            kl_div_history.append(normalized_kl_div.mean().item())
            reward_history.append(total_rewards / GRADIENT_ACCUMULATION_STEPS)

            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS}: Loss: {total_loss / GRADIENT_ACCUMULATION_STEPS} "
                f"Rewards: {total_rewards / GRADIENT_ACCUMULATION_STEPS} "
                f"NegAdv: {neg_advantage} KL: {normalized_kl_div.mean()}"
            )

# After training, plot the metrics
plt.figure(figsize=(12, 5))

# Plot KL divergence
plt.subplot(1, 2, 1)
plt.plot(steps_history, kl_div_history)
plt.xlabel('Steps')
plt.ylabel('KL Divergence')
plt.title('KL Divergence vs Steps')

# Plot rewards
plt.subplot(1, 2, 2)
plt.plot(steps_history, reward_history)
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Steps')

plt.tight_layout()
plt.savefig(f'training_metrics_{KL_FACTOR}.png')
plt.close()

save_directory = f"./checkpoints_{KL_FACTOR}"
model.save_pretrained(save_directory)