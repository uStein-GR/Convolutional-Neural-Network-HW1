import torch
import torch.nn as nn
import numpy as np

def softmax_c(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs1 = softmax_c(x)
print('softmax numpy: ', outputs1)

y = torch.tensor([2.0, 1.0, 0.1])
outputs2 = torch.softmax(y, dim=0)
print('output2: ', outputs2)