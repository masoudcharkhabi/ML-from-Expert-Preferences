from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Replace with Gemma model path or Hugging Face repository name
model_name = "google/gemma-2-2b"  # Adjust this to your Gemma model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# File containing the boolean expressions
input_file = "../data/BIG-Bench-Hard/cot-prompts/boolean_expressions.txt"

print(f"Input file: {input_file}")

def process_file(file_path):
    """Reads the file, extracts task, and processes questions to generate answers."""
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
    for line in lines:
        if line.startswith("Q: "):
            question = line[len("Q: "):].strip()  # Extract the question
            print(f"Processing question: {question}")

            # Create the full prompt by combining the task description and the question
            full_prompt = f"{task_description}\n{question}"
            print(f"Full prompt for Gemma: {full_prompt}")

            # Generate model response
            inputs = tokenizer(full_prompt, return_tensors="pt")
            outputs = model.generate(inputs["input_ids"], max_length=50)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Add question and Gemma's response
            gemma_output_lines.append(f"Q: {question}\n")
            gemma_output_lines.append(f"A: {response}\n\n")
            print(f"Generated answer: {response}")

    # Write the questions and Gemma's answers to a new file
    with open(output_file, "w") as file:
        file.writelines(gemma_output_lines)
    print(f"Gemma's answers written to {output_file}")

# Run the processing function
process_file(input_file)
