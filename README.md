# ML-from-Expert-Preferences

## Problem statement

Background: Active learning boosts model performance by selectively annotating data but suffers from high computational costs, particularly for Large Language Models (LLMs).

Objective: Building on [Bhatt et al. (2024)](https://arxiv.org/abs/2401.06692v3), analyze how varying retraining iterations impacts performance and data selection under a fixed annotation budget in LLMs.

Main Hypothesis: Increasing retraining iterations improves data quality.

## Methods

LLMs: Llama 3.2 1B-Instruct; evaluated with 50K annotation budgets

Treatment Variables: Acquisition strategies (random, entropy, or confidence) and retraining iterations (1 or 2). Additional variations are discussed in the paper.

Random Strategy: Select data points randomly for annotation.

Entropy Strategy: Select data with the highest token prediction entropy.

Confidence Strategy: Select data with the lowest token prediction probabilities.

Retraining Iterations: For $m$ iterations, acquire 50K/$m$ examples per iteration using the specified strategy, retrain the model, and repeat $m$ times.


## Steps to run experiments and view results

The full experiments were run on Google Colab with an A100 GPU on pay-as-you-go credits. Please refer Huggingface [coderGenMC](https://huggingface.co/coderGenMC) and [rnjs1992](https://huggingface.co/rnjs1992) for all model and dataset artifacts and Weights and Biases [mcharkhabi](https://wandb.ai/ai-eval/active-llm?nw=nwusermcharkhabi) or [kwonosubai](https://wandb.ai/ai-eval/active-llm/table?nw=nwuserkwonosubai) for all experiments. The config files for experiments are written to Weights and Biases. Each experiment has a timestamp experiment ID that can be used to connect models, datasets and experiments. The artifacts will be public for 30 days. 

The baseline was built on Gemma and Llamma models with Big-Bench-Hard and Flan V2 datasets. Due to performance and memory management issues we transitioned to abstract implementations on cloud resources. To run the baseline set up on CPU only follow the below instructions. 

1) Install miniconda from https://docs.anaconda.com/free/miniconda/

2) Create a conda environment for the experiments
```
conda create --name MLE-project --file requirements.txt python=3.9 -c conda-forge
```

If you run into issues with installing ray and torch, use the official pytorch Conda channel or install manually after creating the env:
```
conda activate MLE-project
conda install pytorch torchvision torchaudio -c pytorch
pip3 install ray
```

After installing PyTorch, install the remaining dependencies from your requirements.txt (This will install transformers and huggingface_hub):
```
pip3 install -r requirements.txt
```

3) Activate the conda package and environment manager:
```
conda activate MLE-project
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
