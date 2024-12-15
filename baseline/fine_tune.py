import argparse
import logging
import os
import signal
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from config_utils import load_config
from peft import LoraConfig, get_peft_model  # Importing LoRA modules

# Set transformers verbosity to error
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

logging.basicConfig(level=logging.INFO)

# Handle interrupt gracefully
def handle_interrupt(signal: int, frame) -> None:
    """
    Handle keyboard interrupt (CTRL+C) gracefully.

    Args:
        signal (int): Signal number.
        frame: Current stack frame.
    """
    logging.warning("Interrupt received! Shutting down gracefully...")
    exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

def ensure_directory_exists(directory: str, description: str) -> None:
    """
    Ensure that a required directory exists.

    Args:
        directory (str): Path to the directory.
        description (str): Description of the directory.
    """
    if not os.path.isdir(directory):
        logging.error(f"{description} directory {directory} does not exist.")
        exit(1)

def prepare_dataset(data_files: list, tokenizer, max_length: int):
    """
    Prepare dataset from given files by tokenizing the text.

    Args:
        data_files (list): List of file paths to prepare the dataset from.
        tokenizer: Tokenizer instance to tokenize the dataset.
        max_length (int): Maximum length for tokenized sequences.

    Returns:
        Dataset: Tokenized dataset.
    """
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
    
    dataset = load_dataset("text", data_files={"train": data_files})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def main() -> None:
    """
    Main function to fine-tune a language model using specified parameters from a configuration file.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a language model using specified parameters.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (JSON format).",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Ensure output directory exists
    output_dir = config.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        model = AutoModelForCausalLM.from_pretrained(config["model_name"])
        # Set padding token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logging.error(f"Error loading model and tokenizer: {str(e)}")
        exit(1)

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=8,  # Rank of the LoRA update matrices
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Target specific modules in the transformer
        lora_dropout=0.1,  # Dropout rate
        bias="none",  # Bias handling in LoRA
    )
    model = get_peft_model(model, lora_config)

    # Set training device
    device = "cuda" if torch.cuda.is_available() and config.get("device", "gpu").lower() == "gpu" else "cpu"
    model.to(device)
    logging.info(f"Using device: {device}")

    # Prepare dataset
    dataset_file = config.get("flan_file")
    if dataset_file is None:
        bbh_dir = config.get("bbh_dir")
        if bbh_dir is None or not os.path.isdir(bbh_dir):
            logging.error("No valid dataset file or directory specified in the configuration.")
            exit(1)
        dataset_files = [os.path.join(bbh_dir, file_name) for file_name in os.listdir(bbh_dir) if file_name.endswith(".txt")]
    else:
        dataset_files = [dataset_file]

    tokenized_dataset = prepare_dataset(dataset_files, tokenizer, config.get("BBH_MAX_TOKEN_LENGTH", 500))

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_iterations", 1),
        per_device_train_batch_size=config.get("batch_size", 4),
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs",
        report_to="none",  # Disable reporting to WandB by default
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="no",  # Disable evaluation during training by default
        dataloader_num_workers=2,  # Limit number of workers to avoid excessive memory usage
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
    )

    # Fine-tune the model
    logging.info("Starting training...")
    try:
        torch.cuda.empty_cache()  # Clear GPU cache to avoid OOM error
        trainer.train()
        trainer.save_model(output_dir)  # Save the fine-tuned model
    except torch.cuda.OutOfMemoryError:
        logging.error("CUDA out of memory. Try reducing the batch size or using a smaller model.")
        exit(1)
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        exit(1)
    logging.info("Training complete. Model saved to output directory.")

if __name__ == "__main__":
    main()
