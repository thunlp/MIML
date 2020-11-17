## MIML

Source code and dataset for MIML.

### Reqirements:

* Pytorch>=0.4.1
* Python3
* numpy
* tqdm
* boto3
* requests
* regex
* sentencepiece
* sacremoses

OR install with:

> pip install -r requirements.txt


### Data

The data and the data processing program are under the directory `/data`.

### BERT

Following https://github.com/huggingface/transformers, and made some modifications, the modified code is placed under the directory /my_transformers.

#### Pre-trained model:

Load BERT's pre-trained model bert-base-uncased. If the model cannot be downloaded online at runtime due to network reasons, you can pre-download it locally in ./bert-base-uncased path.

### Run:

#### MIML (e.g., 10-way-5-shot)
> python3 main.py --N 10 --K 5 --MI --MF --VAT



