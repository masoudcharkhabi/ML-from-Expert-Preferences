# train.py

class ModelTrainer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.trainer = None
        self.experiment_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f"./models/fine_tuned_model_{self.experiment_id}"
        wandb.init(project="active-llm", name=f"fine_tune_{self.experiment_id}", resume="allow")

    def setup_training(self, train_dataset, eval_dataset=None, tokenizer=None):
        """Set up training arguments and Trainer"""
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch" if eval_dataset is not None else "no",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=0.0005,
            weight_decay=0.01,
            report_to=["wandb"],  # Log training statistics to Weights & Biases
            run_name="model_training"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

    def train_model(self, save_model: bool = True):
        """Train the model"""
        if self.trainer is not None:
            self.trainer.train()
            if save_model:
                # Ensure the output directory exists before saving
                os.makedirs(self.output_dir, exist_ok=True)
                self.trainer.save_model(self.output_dir)
                print(f"Model saved to: {self.output_dir}")
            wandb.finish()
        else:
            raise ValueError("Trainer is not initialized. Please call setup_training first.")
