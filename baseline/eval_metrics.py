import argparse
import logging
import os
import csv
from rouge_score import rouge_scorer
import math
from nltk.translate.bleu_score import sentence_bleu
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)

def evaluate_results(output_dir, ground_truth_dirs):
    """
    Compares generated output with ground truth and calculates metrics.

    Args:
        output_dir (str): Directory containing the generated output files.
        ground_truth_dirs (list): List of directories containing ground truth files.
    """
    def read_file(file_path):
        data = []
        with open(file_path, "r") as f:
            reader = csv.reader(f, delimiter="\t" if file_path.endswith(".tsv") else "\n")
            current_question = ""
            current_answer = ""
            for row in reader:
                if len(row) > 1:
                    # For TSV files
                    current_answer = row[1]
                    data.append((current_question, current_answer))
                elif len(row) == 1:
                    if row[0].startswith("Q: "):
                        current_question = row[0][len("Q: "):].strip()
                    elif row[0].startswith("A: "):
                        current_answer = row[0][len("A: "):].strip()
                        data.append((current_question, current_answer))
        return data

    def get_matching_ground_truth_file(output_file_name):
        base_name = "_".join(output_file_name.split("_")[:-2])  # Get the base file name without model suffix
        for directory in ground_truth_dirs:
            possible_path = os.path.join(directory, f"{base_name}.tsv")
            if os.path.exists(possible_path):
                return possible_path
            possible_path = os.path.join(directory, f"{base_name}.txt")
            if os.path.exists(possible_path):
                return possible_path
        return None

    all_y_true = []
    all_y_pred = []

    for output_file_name in os.listdir(output_dir):
        output_file_path = os.path.join(output_dir, output_file_name)
        ground_truth_file_path = get_matching_ground_truth_file(output_file_name)
        if ground_truth_file_path is None:
            logging.warning(f"No matching ground truth file found for {output_file_name}. Skipping.")
            continue

        output_data = read_file(output_file_path)
        ground_truth_data = read_file(ground_truth_file_path)

        if len(output_data) != len(ground_truth_data):
            logging.warning(f"File lengths do not match for {output_file_name} and {ground_truth_file_path}. Skipping.")
            continue

        y_true = [answer for _, answer in ground_truth_data]
        y_pred = [answer for _, answer in output_data]

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    if all_y_true and all_y_pred:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bleu_scores = []
        edit_distances = []
        for true, pred in zip(all_y_true, all_y_pred):
            # ROUGE Scores
            scores = scorer.score(true, pred)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)
            
            # BLEU Score
            bleu_score = sentence_bleu([true.split()], pred.split())
            bleu_scores.append(bleu_score)
            
            # Edit Distance
            edit_distance = SequenceMatcher(None, true, pred).ratio()
            edit_distances.append(edit_distance)

        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        avg_edit_distance = sum(edit_distances) / len(edit_distances)

        logging.info("Aggregated Performance Metrics:")
        logging.info(f"ROUGE-1 Score: {avg_rouge1:.2f}")
        logging.info(f"ROUGE-2 Score: {avg_rouge2:.2f}")
        logging.info(f"ROUGE-L Score: {avg_rougeL:.2f}")
        logging.info(f"BLEU Score: {avg_bleu:.2f}")
        logging.info(f"Edit Distance Score: {avg_edit_distance:.2f}")
    else:
        logging.info("No data to compare.")


def main():
    """
    Main function to evaluate generated output against ground truth.
    """
    parser = argparse.ArgumentParser(description="Evaluate generated outputs against ground truth.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing the generated output files.",
    )
    parser.add_argument(
        "--ground_truth_dirs",
        type=str,
        nargs='+',
        required=True,
        help="List of directories containing ground truth files.",
    )
    args = parser.parse_args()

    # Evaluate results
    evaluate_results(os.path.join(os.path.dirname(__file__), '..', args.output_dir), [os.path.join(os.path.dirname(__file__), '..', directory) for directory in args.ground_truth_dirs])


if __name__ == "__main__":
    main()
