import argparse
import logging
import os
import csv
import signal
from config_utils import load_config
from model_utils import load_model_and_tokenizer, generate_response

logging.basicConfig(level=logging.INFO)

# Handle interrupt gracefully
def handle_interrupt(signal, frame):
    logging.warning("Interrupt received! Shutting down gracefully...")
    exit(0)

signal.signal(signal.SIGINT, handle_interrupt)


def ensure_directory_exists(directory: str, description: str):
    """
    Ensure that a required directory exists.

    Args:
        directory (str): Path to the directory.
        description (str): Description of the directory.
    """
    if not os.path.isdir(directory):
        logging.error(f"{description} directory {directory} does not exist.")
        exit(1)


def process_bbh_file(file_path: str, output_dir: str, config: dict, tokenizer, model):
    """
    Processes a single BBH file to extract questions and generate answers.

    Args:
        file_path (str): Path to the BBH file.
        output_dir (str): Directory to save output files.
        config (dict): Configuration data.
        tokenizer: Tokenizer instance.
        model: Model instance.
    """
    base_name = os.path.basename(file_path)[:-4]  # Remove '.txt' from the file name
    output_file = os.path.join(output_dir, f"{base_name}_{config['model_name'].replace('/', '_')}.txt")

    logging.info(f"Processing file: {file_path}")
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Extract the task description from line 3 after the "-----"
        separator_index = lines.index("-----\n")
        task_description = lines[separator_index + 1].strip()
        logging.info(f"Task description: {task_description}")

        gemma_output_lines = []
        example_count = 0
        current_question = []
        inside_question = False

        for line in lines:
            if line.startswith("Q: "):
                if inside_question:
                    full_question = "\n".join(current_question).strip()
                    if example_count < config["MAX_EXAMPLES"]:
                        example_count += 1
                        process_single_question(full_question, task_description, config, tokenizer, model, gemma_output_lines)
                    current_question = []  # Reset question buffer
                inside_question = True
                current_question.append(line[len("Q: "):].strip())
            elif line.startswith("A: ") and inside_question:
                full_question = "\n".join(current_question).strip()
                if example_count < config["MAX_EXAMPLES"]:
                    example_count += 1
                    process_single_question(full_question, task_description, config, tokenizer, model, gemma_output_lines)
                current_question = []  # Reset question buffer
                inside_question = False
            elif inside_question:
                current_question.append(line.strip())

        with open(output_file, "w") as out_file:
            out_file.writelines(gemma_output_lines)
        logging.info(f"Answers for {base_name} written to {output_file}")
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")


def process_single_question(full_question: str, task_description: str, config: dict, tokenizer, model, gemma_output_lines: list):
    """
    Process a single question and generate a response.

    Args:
        full_question (str): The full question text.
        task_description (str): Task description to be included with the question.
        config (dict): Configuration data.
        tokenizer: Tokenizer instance.
        model: Model instance.
        gemma_output_lines (list): List to store generated question-answer pairs.
    """
    try:
        full_prompt = f"{task_description}\n{full_question}"
        logging.info(f"Processing question: {full_question}")

        response = generate_response(full_prompt, tokenizer, model, max_length=config["BBH_MAX_TOKEN_LENGTH"])
        gemma_output_lines.append(f"Q: {full_question}\n")
        gemma_output_lines.append(f"A: {response}\n\n")
        logging.info(f"Generated answer: {response}")
    except Exception as e:
        logging.error(f"Error generating response for question: {full_question}. Error: {str(e)}")


def process_flan_file(config: dict, tokenizer, model):
    """
    Processes FLAN data (TSV format) and generates answers for multiple-choice questions.

    Args:
        config (dict): Configuration data.
        tokenizer: Tokenizer instance.
        model: Model instance.
    """
    file_path = config["flan_file"]
    ensure_directory_exists(os.path.dirname(file_path), "FLAN input")

    output_file = os.path.join(config.get("output_dir", "."), f"aqua_train_{config['model_name'].replace('/', '_')}.tsv")

    gemma_output_rows = []
    example_count = 0

    try:
        with open(file_path, "r") as tsvfile:
            reader = csv.reader(tsvfile, delimiter="\t")
            for row in reader:
                if example_count >= config["MAX_EXAMPLES"]:
                    break
                if len(row) < 1:
                    continue
                example_count += 1

                question_with_options = row[0].strip()
                logging.info(f"Processing prompt {example_count}/{config['MAX_EXAMPLES']}: {question_with_options}")

                response = generate_response(question_with_options, tokenizer, model, max_length=config["FLAN_MAX_NEW_TOKENS"])

                answer_key = None
                for option in ["(A)", "(B)", "(C)", "(D)", "(E)"]:
                    if option in response:
                        answer_key = option
                        break

                if not answer_key:
                    logging.warning("No valid answer found in response. Defaulting to (A).")
                    answer_key = "(A)"

                gemma_output_rows.append([question_with_options, answer_key, ""])
                logging.info(f"Generated answer: {answer_key}")

        with open(output_file, "w", newline="") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t")
            writer.writerow(["Prompt", "Answer", "Explanation"])
            writer.writerows(gemma_output_rows)
        logging.info(f"FLAN answers written to {output_file}")
    except Exception as e:
        logging.error(f"Error processing FLAN file {file_path}: {str(e)}")


def main():
    """
    Main function to process BBH or FLAN data based on a configuration file.
    """
    parser = argparse.ArgumentParser(description="Evaluate BBH or FLAN data with a model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (JSON format).",
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=["bbh", "flan"], required=True, help="Data to process: 'bbh' for BBH data or 'flan' for FLAN data."
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(config["model_name"])

    # Process data based on mode
    if args.data == "bbh":
        bbh_dir = config["bbh_dir"]
        ensure_directory_exists(bbh_dir, "BBH")
        output_dir = config.get("output_dir", ".")
        os.makedirs(output_dir, exist_ok=True)

        files_to_process = [os.path.join(bbh_dir, file_name) for file_name in os.listdir(bbh_dir) if file_name.endswith(".txt")]
        for file_path in files_to_process:
            process_bbh_file(file_path, output_dir, config, tokenizer, model)
    elif args.data == "flan":
        process_flan_file(config, tokenizer, model)


if __name__ == "__main__":
    main()
