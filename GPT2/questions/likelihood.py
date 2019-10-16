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

        likelihood = 0.0
        past = None
        text2 = text.transpose(0, 1)
        x = text2[None, 0, None]
        for i in range (len(text2)-1):
            logits, past = model(x, past=past)
            log_softmax = F.log_softmax(logits, dim=3)
            likelihood += log_softmax[-1, -1, -1, text2[i,0]].item()
            x = text2[None, i+1, None]

        return likelihood