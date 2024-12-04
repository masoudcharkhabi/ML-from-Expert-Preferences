# eval.py

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSeq2SeqLM
from evaluate import load
import torch
import wandb
import numpy as np
import datetime
from random import sample

class ModelEvaluator:
    def __init__(self, model_name: str, tokenizer):
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.bleu_metric = load("bleu")
        self.rouge_metric = load("rouge")
        self.experiment_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        wandb.init(project="active-llm", name=f"eval_{self.experiment_id}", resume="allow")

    def evaluate(self, dataset, sample_size=100, batch_size=8):
        """Evaluate model performance on a subset of the dataset using batches"""
        # Sample a subset of the dataset to speed up evaluation
        if len(dataset) > sample_size:
            dataset = sample(list(dataset), sample_size)

        predictions = []
        references = []
        total_loss = 0
        num_tokens = 0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            input_texts = [example["input_text"] for example in batch]
            target_texts = [example["target_text"] for example in batch]

            inputs = self.tokenizer(input_texts, return_tensors="pt", truncation=True, padding="max_length", max_length=100)
            labels = self.tokenizer(target_texts, return_tensors="pt", truncation=True, padding="max_length", max_length=100).input_ids

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=100)
                predicted_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                predictions.extend(predicted_texts)
                references.extend(target_texts)

                # Calculate loss for perplexity in a batch
                output_loss = self.model(**inputs, labels=labels)
                total_loss += output_loss.loss.item() * labels.size(1)
                num_tokens += labels.size(1)

        # Calculate traditional metrics
        accuracy = accuracy_score(references, predictions)
        f1 = f1_score(references, predictions, average='weighted')
        precision = precision_score(references, predictions, average='weighted')
        recall = recall_score(references, predictions, average='weighted')

        # Calculate LLM and NLP specific metrics
        bleu_score = self.bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
        rouge_score = self.rouge_metric.compute(predictions=predictions, references=references)

        # Calculate perplexity
        avg_loss = total_loss / num_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))

        # Log the results to W&B
        wandb.log({
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "bleu": bleu_score,
            "rouge": rouge_score,
            "perplexity": perplexity.item(),
                    })
        wandb.finish()

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "bleu": bleu_score,
            "rouge": rouge_score,
            "perplexity": perplexity.item()
        }
