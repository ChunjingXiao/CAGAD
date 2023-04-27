# CAGAD

This is a repository hosting the code of CAGAD. 

## Data

- For Yelp, Amazon, and PubMed datasets, they will be automatically downloaded when runinng the code.    
- For T-Finance dataset, you can download it from the baseline paper BWGNN: https://proceedings.mlr.press/v162/tang22b.html

## Dependencies

Run the following command to install dependencies with Anaconda virtual environment:
```shell
conda create -n cagad python==3.9
conda activate cagad
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c dglteam dgl-cuda11.3
pip install -r requirements.txt
```

## Run

```shell
# PubMed
python main.py --dataset=pubmed

# T-Finance
python main.py --dataset=tfinance

# Amazon
python main.py --dataset=amazon

# Yelp dataset
python main.py --dataset=yelp --homo=0
```

Description of hyper-parameters can be found in `main.py`.
