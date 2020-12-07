## MIML

Source code for [Meta-Information Guided Meta-Learning for Few-Shot Relation Classification](https://www.aclweb.org/anthology/2020.coling-main.140.pdf).

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

```
pip install -r requirements.txt
```


### Data

The data and the preprocessing code are under the directory `/data`.

### BERT

We build our model based on the BERT implementation of [Huggingface](https://github.com/huggingface/transformers). The code for our model is placed under the directory /my_transformers.

#### Pre-trained model:

Load BERT's pre-trained model bert-base-uncased. If the model cannot be downloaded online at runtime due to network reasons, you can pre-download it locally in ./bert-base-uncased path.

### Run:

#### MIML (e.g., 10-way-5-shot)
```
python3 main.py --N 10 --K 5 --MI --MF --VAT
```

## Cite
If you use the code, please cite this paper:
```
@inproceedings{dong-etal-2020-meta,
    title = "Meta-Information Guided Meta-Learning for Few-Shot Relation Classification",
    author = "Dong, Bowen  and
      Yao, Yuan  and
      Xie, Ruobing  and
      Gao, Tianyu  and
      Han, Xu  and
      Liu, Zhiyuan  and
      Lin, Fen  and
      Lin, Leyu  and
      Sun, Maosong",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.140",
    pages = "1594--1605",
}
```

