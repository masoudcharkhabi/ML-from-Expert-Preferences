from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace with the actual model repository name
model_name = "meta-llama/Meta-Llama-3-8B"

# Download the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model and tokenizer downloaded successfully!")

