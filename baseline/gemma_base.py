import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import csv

BBH_MAX_TOKEN_LENGTH = 500
FLAN_MAX_NEW_TOKENS = 500
MAX_EXAMPLES = 3  # Global parameter to limit the number of examples

# Replace with Gemma model path or Hugging Face repository name
model_name = "google/gemma-2-2b"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# File paths for BBH and FLAN datasets
bbh_dir = "../data/BIG-Bench-Hard/cot-prompts/"
flan_file = "../data/flan/v2/cot_data/aqua_train.tsv"

def process_bbh_file(file_path):
    """Processes BBH data to extract questions and generate answers."""
    # Determine the output file name
    file_prefix, file_extension = os.path.splitext(file_path)
    output_file = f"{file_prefix}_gemma{file_extension}"

    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract the task description from line 3 after the "-----"
    separator_index = lines.index("-----\n")
    task_description = lines[separator_index + 1].strip()
    print(f"Task description: {task_description}")

    # Process each line and handle "Q: " questions
    gemma_output_lines = []
    example_count = 0  # Counter to track processed examples
    for line in lines:
        if line.startswith("Q: "):
            if example_count >= MAX_EXAMPLES:
                break  # Stop processing if the max number of examples is reached
            example_count += 1

            question = line[len("Q: "):].strip()  # Extract the question
            print(f"Processing question: {question}")

            # Create the full prompt by combining the task description and the question
            full_prompt = f"{task_description}\n{question}"
            print(f"Full prompt for Gemma: {full_prompt}")

            # Generate model response
            inputs = tokenizer(full_prompt, return_tensors="pt")
            outputs = model.generate(inputs["input_ids"], max_length=BBH_MAX_TOKEN_LENGTH)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Add question and Gemma's response
            gemma_output_lines.append(f"Q: {question}\n")
            gemma_output_lines.append(f"A: {response}\n\n")
            print(f"Generated answer: {response}")

    # Write the questions and Gemma's answers to a new file
    with open(output_file, "w") as file:
        file.writelines(gemma_output_lines)
    print(f"Gemma's answers written to {output_file}")

def process_flan_file(file_path):
    """Processes FLAN data (TSV format) and generates answers for multiple-choice questions."""
    # Determine the output file name
    file_prefix, file_extension = os.path.splitext(file_path)
    output_file = f"{file_prefix}_gemma.tsv"

    # Prepare for TSV output
    gemma_output_rows = []
    example_count = 0  # Counter to track processed examples

    # Read the TSV file
    with open(file_path, "r") as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")
        for row in reader:
            if example_count >= MAX_EXAMPLES:
                break  # Stop processing if the max number of examples is reached
            if len(row) < 1:  # Ensure there is at least one column for the question
                continue
            example_count += 1

            # Extract the question and options from the first column
            question_with_options = row[0].strip()
            print(f"Processing prompt: {question_with_options}")

            # Generate model response
            inputs = tokenizer(question_with_options, return_tensors="pt")
            outputs = model.generate(inputs["input_ids"], max_new_tokens=FLAN_MAX_NEW_TOKENS)  # Use max_new_tokens
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the answer key from the response
            # Ensure Gemma outputs a valid choice: (A), (B), (C), (D), or (E)
            answer_key = None
            for option in ["(A)", "(B)", "(C)", "(D)", "(E)"]:
                if option in response:
                    answer_key = option
                    break

            # If no valid answer is found, default to "(A)"
            if not answer_key:
                print(f"Warning: No valid answer found in response. Defaulting to (A).")
                answer_key = "(A)"

            # Add the question, answer, and explanation to the output rows
            gemma_output_rows.append([question_with_options, answer_key, ""])

            print(f"Generated answer: {answer_key}")

    # Write the questions and Gemma's answers to a new TSV file
    with open(output_file, "w", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t")
        # Write header
        writer.writerow(["Prompt", "Answer", "Explanation"])
        # Write data rows
        writer.writerows(gemma_output_rows)

    print(f"Gemma's answers written to {output_file}")

def main():
    """Main function to process BBH or FLAN data based on command-line arguments."""
    parser = argparse.ArgumentParser(description="Process BBH or FLAN data using Gemma.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["bbh", "flan"],
        required=True,
        help="Mode of operation: 'bbh' for BBH data or 'flan' for FLAN data.",
    )
    args = parser.parse_args()

    if args.mode == "bbh":
        # Specify the BBH input file
        bbh_file = os.path.join(bbh_dir, "boolean_expressions.txt")
        print(f"Processing BBH file: {bbh_file}")
        process_bbh_file(bbh_file)
    elif args.mode == "flan":
        print(f"Processing FLAN file: {flan_file}")
        process_flan_file(flan_file)

# Run the program
if __name__ == "__main__":
    main()
