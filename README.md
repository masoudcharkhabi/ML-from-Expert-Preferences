# ML-from-Expert-Preferences

## Proposal

### Goal
Active learning seeks to optimize the selection of data points for annotation, improving model performance efficiently while minimizing the need for large labeled datasets. However, two significant challenges hinder its application: the cost of annotation and the computational burden of iteratively selecting data points and retraining the model. To address these issues, [Bhatt et al. (2024)](https://arxiv.org/abs/2401.06692v3) propose "experimental design," which involves selecting all examples for annotation based on an initial model and performing a single round of fine-tuning. While this approach reduces computational overhead, their work does not fully explore the trade-off between performance and computational costs, nor does it directly compare experimental design with traditional active learning.

This project builds upon [Bhatt et al. (2024)](https://arxiv.org/abs/2401.06692v3) by investigating how performance and the selection of data points for annotation evolve as we vary the number of retraining iterations while keeping the annotation budget fixed. We hypothesize that additional retraining increases the diversity of annotated examples, which, in turn, improves active learning performance.

## Steps to run experiments

1) Install miniconda from https://docs.anaconda.com/free/miniconda/

2) Create a conda environment for the experiments
```
conda create --name cs329h-project --file requirements.txt python=3.9
```

If you run into issues with torch install, use the official pytorch Conda channel to install torch:
```
conda install pytorch torchvision torchaudio -c pytorch
conda activate cs329h-project
```

After installing PyTorch, install the remaining dependencies from your requirements.txt (This will install transformers and huggingface_hub):
```
pip3 install -r requirements.txt
```

3) Activate the conda package and environment manager:
```
conda activate cs329h-project
```

4) Run eval.py with a selected model and chain-of-thought data prompt flag. The results will be written to a file with the matching prefix and _model_name postfix.
```
python3 eval.py --config config_gemma.json --mode bbh
```
