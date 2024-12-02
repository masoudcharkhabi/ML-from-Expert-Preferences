import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import csv

def load_config(config_file):
    """Load configuration from a JSON file."""
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file {config_file} not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Configuration file {config_file} is not a valid JSON.")
        exit(1)

def ensure_directory_exists(directory, description):
    """Ensure that a required directory exists."""
    if not os.path.isdir(directory):
        print(f"Error: {description} directory {directory} does not exist.")
        exit(1)

def process_bbh_directory(config, tokenizer, model):
    """Processes all .txt files in the BBH directory to extract questions and generate answers."""
    bbh_dir = config["bbh_dir"]
    ensure_directory_exists(bbh_dir, "BBH")

    output_dir = config.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all .txt files in the directory
    for file_name in os.listdir(bbh_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(bbh_dir, file_name)
            # Correct the output file name
            base_name = file_name[:-4]  # Remove '.txt' from the file name
            output_file = os.path.join(output_dir, f"{base_name}_{config['model_name'].replace('/', '_')}.txt")

            print(f"Processing file: {file_path}")
            with open(file_path, "r") as file:
                lines = file.readlines()

            # Extract the task description from line 3 after the "-----"
            try:
                separator_index = lines.index("-----\n")
                task_description = lines[separator_index + 1].strip()
                print(f"Task description: {task_description}")
            except ValueError:
                print(f"Error: File {file_path} is not in the expected format (missing '-----'). Skipping.")
                continue

            # Process questions and answers
            gemma_output_lines = []
            example_count = 0
            current_question = []
            inside_question = False

            for line in lines:
                # Start capturing question if line starts with "Q: "
                if line.startswith("Q: "):
                    if inside_question:  # If we're already capturing, finalize the previous question
                        full_question = "\n".join(current_question).strip()
                        if example_count < config["MAX_EXAMPLES"]:
                            example_count += 1
                            print(f"Processing question {example_count}/{config['MAX_EXAMPLES']}: {full_question}")
                            full_prompt = f"{task_description}\n{full_question}"

                            # Generate model response
                            inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
                            if tokenizer.pad_token_id is None:
                                tokenizer.pad_token_id = tokenizer.eos_token_id

                            outputs = model.generate(
                                inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                max_length=config["BBH_MAX_TOKEN_LENGTH"],
                            )
                            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            gemma_output_lines.append(f"Q: {full_question}\n")
                            gemma_output_lines.append(f"A: {response}\n\n")
                            print(f"Generated answer: {response}")
                        else:
                            break
                        current_question = []  # Reset question buffer
                    inside_question = True
                    current_question.append(line[len("Q: "):].strip())
                # Stop capturing question and finalize on "A: "
                elif line.startswith("A: ") and inside_question:
                    full_question = "\n".join(current_question).strip()
                    if example_count < config["MAX_EXAMPLES"]:
                        example_count += 1
                        print(f"Processing question {example_count}/{config['MAX_EXAMPLES']}: {full_question}")
                        full_prompt = f"{task_description}\n{full_question}"

                        # Generate model response
                        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
                        if tokenizer.pad_token_id is None:
                            tokenizer.pad_token_id = tokenizer.eos_token_id

                        outputs = model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_length=config["BBH_MAX_TOKEN_LENGTH"],
                        )
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        gemma_output_lines.append(f"Q: {full_question}\n")
                        gemma_output_lines.append(f"A: {response}\n\n")
                        print(f"Generated answer: {response}")
                    else:
                        break
                    current_question = []  # Reset question buffer
                    inside_question = False
                # Accumulate question lines
                elif inside_question:
                    current_question.append(line.strip())

            # Write output for this file
            with open(output_file, "w") as out_file:
                out_file.writelines(gemma_output_lines)
            print(f"Answers for {file_name} written to {output_file}")

def process_flan_file(config, tokenizer, model):
    """Processes FLAN data (TSV format) and generates answers for multiple-choice questions."""
    file_path = config["flan_file"]
    ensure_directory_exists(os.path.dirname(file_path), "FLAN input")

    output_file = os.path.join(config.get("output_dir", "."), f"aqua_train_{config['model_name'].replace('/', '_')}.tsv")

    gemma_output_rows = []
    example_count = 0

    with open(file_path, "r") as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")
        for row in reader:
            if example_count >= config["MAX_EXAMPLES"]:
                break
            if len(row) < 1:
                continue
            example_count += 1

            question_with_options = row[0].strip()
            print(f"Processing prompt {example_count}/{config['MAX_EXAMPLES']}: {question_with_options}")

            inputs = tokenizer(question_with_options, return_tensors="pt", padding=True, truncation=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad_token_id to eos_token_id if not already set

            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],  # Explicitly pass the attention mask
                max_new_tokens=config["FLAN_MAX_NEW_TOKENS"],
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            answer_key = None
            for option in ["(A)", "(B)", "(C)", "(D)", "(E)"]:
                if option in response:
                    answer_key = option
                    break

            if not answer_key:
                print("Warning: No valid answer found in response. Defaulting to (A).")
                answer_key = "(A)"

            gemma_output_rows.append([question_with_options, answer_key, ""])
            print(f"Generated answer: {answer_key}")

    with open(output_file, "w", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t")
        writer.writerow(["Prompt", "Answer", "Explanation"])
        writer.writerows(gemma_output_rows)
    print(f"FLAN answers written to {output_file}")

def main():
    """Main function to process BBH or FLAN data based on a configuration file."""
    parser = argparse.ArgumentParser(description="Evaluate BBH or FLAN data with a model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (JSON format).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["bbh", "flan"],
        required=True,
        help="Mode of operation: 'bbh' for BBH data or 'flan' for FLAN data.",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])

    # Handle the pad token issue by assigning `pad_token` to `eos_token` if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Process data based on mode
    if args.mode == "bbh":
        process_bbh_directory(config, tokenizer, model)
    elif args.mode == "flan":
        process_flan_file(config, tokenizer, model)

if __name__ == "__main__":
    main()
