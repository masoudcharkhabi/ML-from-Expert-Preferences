# eval.py
import random
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from transformers import logging as transformers_logging
import torch
import wandb
import datetime
from evaluate import load

class MMLUMixedDatasetLoader:
    def __init__(self, loaded_config, n_examples, subject_config='all'):
        """
        Initialize the MMLU mixed dataset loader.

        Args:
            n_examples (int): Number of random examples to include in the mixed dataset.
            subject_config (str): Subject configuration to load from MMLU (e.g., 'abstract_algebra', 'all', etc.).
        """
        self.n_examples = n_examples
        self.subject_config = subject_config
        self.mixed_dataset = None
        self.loaded_config = loaded_config

    def load_mmlu_mixed_dataset(self):
        """
        Load and create a mixed dataset from different MMLU subjects on Hugging Face.

        Returns:
            Dataset: A mixed Hugging Face Dataset object with random examples from different MMLU subjects.
        """
        # Load the MMLU dataset from Hugging Face with the specified subject configuration
        mmlu_dataset = load_dataset(self.loaded_config["hf_eval_dataset_name"], self.subject_config)

        # Select N random examples from train, validation, and test sets if available
        all_examples = []
        if "train" in mmlu_dataset:
            all_examples.extend(mmlu_dataset["train"])
        if "validation" in mmlu_dataset:
            all_examples.extend(mmlu_dataset["validation"])
        if "test" in mmlu_dataset:
            all_examples.extend(mmlu_dataset["test"])

        # Select N random examples
        random.seed(42)  # Set seed for reproducibility
        selected_examples = random.sample(all_examples, min(self.n_examples, len(all_examples)))

        # Extract the question, choices, and answer for each example
        prompts = []
        responses = []
        choices_labels = ["A", "B", "C", "D"]
        for example in selected_examples:
            prompt = example["question"]
            for idx, choice_label in enumerate(choices_labels):
                choice_key = f"choice{idx}"
                if choice_key in example:
                    prompt += f"\n{choice_label}. {example[choice_key]}"
            prompts.append(prompt)
            responses.append(choices_labels[example["answer"]])

        # Create a Hugging Face Dataset object
        self.mixed_dataset = Dataset.from_dict({
            "instruction": prompts,
            "response": responses
        })

    def save_to_parquet(self, output_path="mmlu_mixed_dataset.parquet"):
        """
        Save the mixed dataset to a Parquet file.

        Args:
            output_path (str): The file path to save the Parquet file.
        """
        if self.mixed_dataset is None:
            raise ValueError("Dataset is not loaded. Please run load_mmlu_mixed_dataset() first.")

        # Convert dataset to Apache Arrow Table and save as Parquet
        arrow_table = pa.Table.from_pandas(self.mixed_dataset.to_pandas())
        pq.write_table(arrow_table, output_path)
        print(f"Dataset saved to {output_path}")

class ModelEvaluator:
    def __init__(self, model_name: str, tokenizer, sample_size, max_length, loaded_config):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.max_length = max_length
        self.rouge_metric = load("rouge")
        self.accuracy_metric = load("accuracy")
        self.experiment_id = loaded_config['experiment_id']
        self.max_new_tokens = loaded_config['max_new_tokens']
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Assign the pad_token to eos_token to avoid the padding warning
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set pad_token_id to eos_token_id explicitly in the model configuration
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Initialize Weights & Biases with increased timeout
        wandb.init(
            project="active-llm",
            name=f"eval_{self.experiment_id}",
            resume="allow",
            settings=wandb.Settings(init_timeout=240)
        )

    def evaluate(self, dataset, batch_size=8, debug=False):
        """Evaluate model performance on a subset of the dataset using batches"""
        # Sample a subset of the dataset to speed up evaluation
        if len(dataset) > self.sample_size:
            dataset = dataset.shuffle(seed=42).select(range(min(self.sample_size, len(dataset))))

        predictions = []
        references = []
        total_loss = 0
        num_tokens = 0
        correct_predictions = 0

        # Function to extract labels from generated text
        def extract_label_from_prediction(prediction: str):
            """
            Extract the first valid label from the last three tokens of the prediction.

            Args:
                prediction (str): The text generated by the model.

            Returns:
                str: The extracted label (A, B, C, D) or 'Unknown' if no valid label is found.
            """
            options = ["A", "B", "C", "D"]  # Valid options

            # Tokenize the prediction into words/tokens
            tokens = prediction.strip().split()  # Split on whitespace

            # Check the last three tokens for valid options
            for token in tokens[-3:]:  # Look at the last three tokens
                if token.upper() in options:  # Case-insensitive comparison
                    return token.upper()  # Return the valid option in uppercase

            return "Unknown"  # Return "Unknown" if no valid label is found

        for i in range(0, len(dataset), batch_size):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            input_texts = batch["instruction"]
            target_texts = batch["response"]

            # Add explicit instructions to the input text for prompt engineering
            modified_input_texts = [
                f"{input_text}\n\n"
                "Please answer with one of the following options: A, B, C, or D.\n"
                "Do NOT repeat the question.\n"
                "Only output a single character as the final answer.\n"
                "For example, if the answer is option B, the output should be: B"
                "\n\n"
                for input_text in input_texts
            ]

            # Tokenize inputs and labels
            inputs = self.tokenizer(modified_input_texts, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length)
            labels = self.tokenizer(target_texts, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length).input_ids

            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)

            with torch.no_grad():
                # Generate outputs using max_new_tokens to limit response length
                outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                raw_predictions = [self.tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]

                # Post-process predictions to extract labels
                predicted_texts = [extract_label_from_prediction(pred) for pred in raw_predictions]

                predictions.extend(predicted_texts)
                references.extend(target_texts)
                correct_predictions += sum([1 if pred == ref else 0 for pred, ref in zip(predicted_texts,
                                                                                         target_texts)])

                # Calculate loss for perplexity in a batch
                output_loss = self.model(**inputs, labels=labels)
                total_loss += output_loss.loss.item() * labels.size(1)
                num_tokens += labels.size(1)

                if debug:
                    # Debugging: print mismatches to understand why accuracy is low
                    for idx, (input_text,
                              raw_pred,
                              processed_pred,
                              target_text) in enumerate(zip(modified_input_texts,
                                                            raw_predictions,
                                                            predicted_texts,
                                                            target_texts)):
                        if processed_pred != target_text:
                            print(f"Mismatch at batch {i}, example {idx}:")
                            print(f"Input: {input_text}")
                            print(f"Raw Prediction: {raw_pred}")
                            print(f"Processed Prediction: {processed_pred}")
                            print(f"Target: {target_text}")

                        # if processed_pred == target_text:
                        #     print(f"Match at batch {i}, example {idx}:")
                        #     print(f"Input: {input_text}")
                        #     print(f"Raw Prediction: {raw_pred}")
                        #     print(f"Processed Prediction: {processed_pred}")
                        #     print(f"Target: {target_text}")

                        # Log to wandb for deeper inspection
                        wandb.log({
                            "input_text": input_text,
                            "raw_prediction": raw_pred,
                            "processed_prediction": processed_pred,
                            "target_text": target_text,
                        })

        # Calculate metrics
        accuracy = correct_predictions / len(dataset)
        rouge_score = self.rouge_metric.compute(predictions=predictions, references=references)
        avg_loss = total_loss / num_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))

        # Log the results to W&B
        wandb.log({
            "accuracy": accuracy,
            "rouge": rouge_score,
            "perplexity": perplexity.item(),
            "eval_examples": len(dataset)
        })
        wandb.finish()

        return {
            "accuracy": accuracy,
            "rouge": rouge_score,
            "perplexity": perplexity.item(),
            "eval_examples": len(dataset)
        }
