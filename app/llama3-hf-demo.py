import transformers
import torch
from transformers import AutoTokenizer

# model = "meta-llama/Llama3.3-70B-Instruct"

model = "meta-llama/Llama-3.3-70B-Instruct"

#tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'I have tomatoes, basil and cheese at home. What can I cook for dinner?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    #eos_token_id=tokenizer.eos_token_id,
    truncation = True,
    max_length=400,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")



""" import transformers
import torch

model_id = "meta-llama/Llama-3.3-70B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1]) """
