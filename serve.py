# serve.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification # Import necessary classes

class ModelServer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Load the correct model type for classification
        # self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        """Set the tokenizer for the model"""
        self.tokenizer = tokenizer

    def run_inference(self, input_text: str):
        """Generate output for a given input text"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Please use set_tokenizer method.")

        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        # For sequence classification, get the logits and predict the class
        outputs = self.model(**inputs).logits
        predicted_class_id = outputs.argmax().item()
        # If you have the labels, you can map the ID to a label
        # predicted_label = model.config.id2label[predicted_class_id]
        return predicted_class_id # or predicted_label if you have the labels

    def store_output(self, input_text: str, output_path: str):
        """Store the generated output in a file"""
        output = self.run_inference(input_text)
        with open(output_path, "w") as file:
            file.write(str(output)) # Write the predicted class ID to the file