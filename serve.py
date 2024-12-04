# serve.py

import torch

class ModelServer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        """Set the tokenizer for the model"""
        self.tokenizer = tokenizer

    def run_inference(self, input_text: str):
        """Generate output for a given input text"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Please use set_tokenizer method.")

        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def store_output(self, input_text: str, output_path: str):
        """Store the generated output in a file"""
        output = self.run_inference(input_text)
        with open(output_path, "w") as file:
            file.write(output)

