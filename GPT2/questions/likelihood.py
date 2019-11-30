import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def log_likelihood(model, text):
    """
    Compute the log-likelihoods for a string `text`
    :param model: The GPT-2 model
    :param texts: A tensor of shape (1, T), where T is the length of the text
    :return: The log-likelihood. It should be a Python scalar. 
        NOTE: for simplicity, you can ignore the likelihood of the first token in `text`.
    """

    with torch.no_grad():
        ## TODO:
        ##  1) Compute the logits from `model`;
        ##  2) Return the log-likelihood of the `text` string. It should be a Python scalar.
        ##      NOTE: for simplicity, you can ignore the likelihood of the first token in `text`

        # we use cross entropy loss to measure distance of two probability group.
        # since this is nearly a classfication problem, we are actualy
        # compare the predict probability [0.2, 0.3, 0.0, 0.5] with class category [0, 0, 1, 0, 0 ...].
        # just remember, here are all probabilities.
        logits = model(text, past=None)
        loss = nn.CrossEntropyLoss(reduction='sum')
        likelihood = -loss(logits[0][:-1], text[0][1:]).item()
        return likelihood
