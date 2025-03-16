from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./checkpoints")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

prompt = "Once upon a time there was"

model.to("cuda")
model.eval()

input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

output = model.generate(input_ids, max_length=500, do_sample=True)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print()
print(output_text)