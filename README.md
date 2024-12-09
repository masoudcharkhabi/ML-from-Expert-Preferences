# ML-from-Expert-Preferences

## Proposal

### Goal

Background: Active learning boosts model performance by selectively annotating data but suffers from high computational costs, particularly for Large Language Models (LLMs).
Objective: Building on[Bhatt et al. (2024)](https://arxiv.org/abs/2401.06692v3), analyze how varying retraining iterations impacts performance and data selection under a fixed annotation budget.
Main Hypothesis: Increasing retraining iterations improves data quality.

## Steps to run experiments

The full experiments were run on Google Colab with an A100 GPU on pay-as-you-go credits. Please refer Huggingface [coderGenMC](https://huggingface.co/coderGenMC) and [rnjs199](https://huggingface.co/rnjs199) for all model and dataset artifacts and Weights and Biases [mcharkhabi](https://wandb.ai/ai-eval/active-llm?nw=nwusermcharkhabi) or [kwonosubai](https://wandb.ai/ai-eval/active-llm/table?nw=nwuserkwonosubai) for all experiments. The config files for experiments are written to Weights and Biases. The artifacts will be public for 30 days. 

To run the baseline set up on CPU only follow the below instructions. 

1) Install miniconda from https://docs.anaconda.com/free/miniconda/

2) Create a conda environment for the experiments
```
conda create --name cs329h-project --file requirements.txt python=3.9 -c conda-forge
```

If you run into issues with installing ray and torch, use the official pytorch Conda channel or install manually after creating the env:
```
conda activate cs329h-project
conda install pytorch torchvision torchaudio -c pytorch
pip3 install ray

```

After installing PyTorch, install the remaining dependencies from your requirements.txt (This will install transformers and huggingface_hub):
```
pip3 install -r requirements.txt
```

3) Activate the conda package and environment manager:
```
conda activate cs329h-project
```

4) Run inference.py with a selected model and chain-of-thought data prompt flag. The results will be written to a file with the matching prefix and _model_name postfix.
```
python3 inference.py --config config_gemma.json --data bbh
```

5) Run eval_metrics.py to compare generated output in data/output to ground truth
```
python3 eval_metrics.py --output_dir data/output --ground_truth_dirs data/BIG-Bench-Hard/cot-prompts data/flan/v2/cot_data
```

6) To run the unittests run in the baseline directory
```
pytest test_eval.py
```

7) For a compatible implementation with Google Colab use the colab branch
```
git checkout -b colab
```
