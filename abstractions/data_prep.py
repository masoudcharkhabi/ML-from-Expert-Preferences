# data_prep.py

from datasets import load_dataset

class DataPreparation:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = None

    def load_data(self):
        """Load dataset from Hugging Face"""
        self.dataset = load_dataset(self.dataset_path)
        return self.dataset

    def preprocess(self, example):
        """Preprocess dataset into input-output pairs"""
        return {
            "input_text": example['inputs'],
            "target_text": example['targets'],
        }

    def tokenize_function(self, examples, tokenizer):
        """Tokenize input and output text"""
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            examples["target_text"],
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def prepare_dataset(self, tokenizer):
        """Prepare the dataset for training"""
        train_dataset = self.dataset["train"].map(self.preprocess)
        train_dataset = train_dataset.map(lambda x: self.tokenize_function(x, tokenizer), batched=True)
        eval_dataset = self.dataset["validation"].map(self.preprocess) if "validation" in self.dataset else None
        eval_dataset = eval_dataset.map(lambda x: self.tokenize_function(x, tokenizer), batched=True) if eval_dataset else None
        # Return only train_dataset instead of a tuple
        return train_dataset 

    def dataset_info(self):
        """Print information about the dataset, such as the size"""
        if self.dataset:
            for split in self.dataset.keys():
                print(f"Split: {split}, Number of examples: {len(self.dataset[split])}")
        else:
            print("Dataset is not loaded. Please call load_data() first.")        