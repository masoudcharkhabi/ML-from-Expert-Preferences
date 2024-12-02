# ML-from-Expert-Preferences

## Proposal

TODO

## Steps to run experiments

1) Install miniconda from https://docs.anaconda.com/free/miniconda/

2) Create a conda environment for the experiments
```
conda create --name cs329h-project --file requirements.txt python=3.9
```

Use the official pytorch Conda channel to install torch:

```
conda create --name cs329h-project python=3.9
conda activate cs329h-project
```

Install PyTorch
```
conda install pytorch torchvision torchaudio -c pytorch
```

After installing PyTorch, install the remaining dependencies from your requirements.txt (This will install transformers and huggingface_hub):
```
pip3 install -r requirements.txt
```

3) Activate the conda package and environment manager:
```
conda activate cs329h-project
```

4) Run the basic input and output script baseline/gemma_base.py
```
python3 gemma_base.py
```

5) The results of gemma will be written to a .txt file with the matching prefix and _gemma postfix.