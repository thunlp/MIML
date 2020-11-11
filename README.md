### Computing infrastructure

#### OS:

Distributor ID: Ubuntu
Description:    Ubuntu 16.04.1 LTS
Release:    16.04

#### GPU:

GeForce RTX 2080Ti

#### Language:

Python 3.7

#### Required packages:

```
torch
numpy
# for transformers
tqdm
boto3
requests
regex
sentencepiece
sacremoses
```

OR install with:

> pip install -r requirements.txt


### Hyperparameters:

The method of choosing hyperparameters:

Grid Search

### Pre-processing:

The pre-processed files for meta-information are located at ./data/meta-info.json

### Pre-trained model:

Load bert's pre-trained model bert-base-uncased. If the model cannot be downloaded online at runtime due to network reasons, you can pre-download it locally in ./bert-base-uncased path.

### Run:

#### MIML (e.g., 10-way-5-shot)
> python3 main.py --N 10 --K 5 --MI --MF --VAT



