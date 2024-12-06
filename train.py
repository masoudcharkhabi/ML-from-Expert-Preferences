# train.py
import os
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import wandb

class ModelTrainer:
    def __init__(self, model_name: str, experiment_id: str, loaded_config):
        self.model_name = model_name
        self.experiment_id = experiment_id
        self.loaded_config = loaded_config
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            offload_folder="./offload"  # Folder to store offloaded parts of the model
        )
        self.trainer = None
        self.output_dir = f"./models/fine_tuned_model_{self.experiment_id}"
        wandb.init(project="active-llm", name=f"fine_tune_{self.experiment_id}", resume="allow")

    def setup_training(self, train_dataset, eval_dataset=None, tokenizer=None):
        """Set up training arguments and Trainer"""

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Adjust the tokenizer to a reduced max length to reduce memory
        train_dataset = train_dataset.map(
            lambda examples: tokenizer(examples['input_text'], truncation=True, padding='max_length', max_length=32),
            batched=True
        )

        # Training arguments without DeepSpeed and offloading
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch" if eval_dataset is not None else "no",
            learning_rate=2e-5,
            per_device_train_batch_size=1,  # Reduce batch size to avoid memory issues
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,  # Reduce gradient accumulation to lower memory needs
            # batch_size=50, # TODO: re-evaluate at 1 epoch
            optim="adamw_torch",
            lr_scheduler_type="linear",
            warmup_ratio=0.03,
            # num_train_epochs=0.001,
            num_train_epochs=self.loaded_config['num_train_epochs'],
            weight_decay=0,
            report_to=["wandb"],
            run_name="model_training",
            fp16=False,  # Disable mixed precision to avoid potential issues
            gradient_checkpointing=False,  # Disable gradient checkpointing to prevent increased memory usage during backpropagation
            seed=42,
            logging_steps=50,      # Log metrics to wandb every 100 steps
            save_strategy="steps", 
            save_steps=10000,      # Save a checkpoint every 500 steps
            save_total_limit=4     # Keep only the 3 most recent checkpoints
        )

        # Use DataCollatorForLanguageModeling for data handling
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

    def train_model(self, save_model: bool = True):
        """Train the model"""
        if self.trainer is not None:
            try:
                self.trainer.train()
                if save_model:
                    # Ensure the output directory exists before saving
                    os.makedirs(self.output_dir, exist_ok=True)
                    self.trainer.save_model(self.output_dir)
                    print(f"Model saved to: {self.output_dir}")
            except RuntimeError as e:
                print("RuntimeError occurred:", e)
            finally:
                wandb.finish()
        else:
            raise ValueError("Trainer is not set up. Please call setup_training() first.")
