## WordPiece tokenization

WordPiece is the tokenization algorithm Google developed to pretrain BERT. It has since been reused in quite a few Transformer models based on BERT, such as DistilBERT, MobileBERT, Funnel Transformers, and MPNET. It’s very similar to BPE in terms of the training, but the actual tokenization is done differently.

## How it works?
the diagram below shows the steps to fit the model based on a corpus. 
![Screenshot](Data\wordpiece_diagram.png)


## Usage

```python 
from WordPiece import wordpiece
corpus ='''
        Object raspberrypi functools dict kwargs. Gevent raspberrypi functools. Dunder raspberrypi decorator dict didn't lambda zip import pyramid, she lambda iterate?
        Kwargs raspberrypi diversity unit object gevent. Import fall integration decorator unit django yield functools twisted. Dunder integration decorator he she future.
        Python raspberrypi community pypy. Kwargs integration beautiful test reduce gil python closure. Gevent he integration generator fall test kwargs raise didn't visor he itertools...
        Reduce integration coroutine bdfl he python. Cython didn't integration while beautiful list python didn't nit!
        Object fall diversity 2to3 dunder script. Python fall for: integration exception dict kwargs dunder pycon. Import raspberrypi beautiful test import six web. Future 
        integration mercurial self script web. Return raspberrypi community test she stable.
        Django raspberrypi mercurial unit import yield raspberrypi visual rocksdahouse. Dunder raspberrypi mercurial list reduce class test scipy helmet zip?
    '''
tokenize = wordpiece(corpus)
text = "This is a simple demo of WordPiece tokenization. exception! "
print(tokenize.tokenize(text))
```
# Instructions

To be able to run all the commands in your local machine start by creating a virtual env by runing : python -m venv .venv

- To activate the virtual environment: `./console.bat`.
- To install dependencies: `./install.bat`
- To register added dependencies: `./freeze.bat`