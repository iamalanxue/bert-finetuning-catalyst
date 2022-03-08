import os
import warnings
import logging
from typing import Mapping, List
from pprint import pprint

# Numpy and Pandas 
import numpy as np
import pandas as pd

# PyTorch 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Transformers 
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Catalyst
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, F1ScoreCallback, OptimizerCallback
from catalyst.dl.callbacks import CheckpointCallback, InferCallback
from catalyst.utils import set_global_seed, prepare_cudnn

from model import BertForSequenceClassification

from data import TextClassificationDataset
# model = BertForSequenceClassification("bert-base-uncased")

# checkpoint = torch.load("logdir/checkpoints/train.2.pth")
# model.load_state_dict(torch.load('logdir/checkpoints/train.2.pth')['state_dict'])
# model.eval()

MODEL_NAME = 'distilbert-base-uncased'

test_df = pd.read_csv("data/motions_laws/test.csv").fillna('')


test_dataset = TextClassificationDataset(
    texts=test_df['description'].values.tolist(),
    labels=None,
    label_dict=None,
    max_seq_length=32,
    model_name=MODEL_NAME
)


test_loaders = {
    "test": DataLoader(dataset=test_dataset,
                        batch_size=16, 
                        shuffle=False) 
}

model = BertForSequenceClassification(pretrained_model_name=MODEL_NAME,
                                            num_classes=2)

runner = SupervisedRunner(
    input_key=(
        "features",
        "attention_mask"
    )
)

runner.infer(
    model=model,
    loaders=test_loaders,
    callbacks=[
        CheckpointCallback(
            resume="logdir/checkpoints/best.pth"
        ),
        InferCallback(),
    ],   
    verbose=True
)


predicted_probs = runner.callbacks[0].predictions['logits']
predictions = []






#convert predictions from confidence to T, F
for pred in predicted_probs:
    if pred[0] > pred[1]:
        predictions.append('F')
    else:
        predictions.append('T')



### Calculate accuracy on test set
import csv

filename = 'data/motions_laws/test.csv'
# print(len(predictions))


with open(filename, 'r') as csvfile:
    datareader = csv.reader(csvfile)

    c=0
    correct=0
    first = True
    for row in datareader:
        if first:
            first=False
        else:
            if row[4] == predictions[c]:
                correct += 1
            c+=1

print(correct/c)
        
