from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace with Gemma model path
model_name = "google/gemma-2-2b"  # Local path or Hugging Face repo name if using API

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example input-output behavior
input_text = "What is the meaning of life?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=50)

print("Input:", input_text)
print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

