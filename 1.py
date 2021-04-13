import torch
import numpy as np

a = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])

b = a[[0, 0, 0, 1, 2]]
print(b)

a = torch.tensor([0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 4, 5, 5, 6, 6, 7, 8])
b = []
index = -1
ori = a[0]
for i in range(len(a)):
    if a[i] == ori:
        index += 1
        b.append(index)
    else:
        ori = a[i]
        index = 0
        b.append(index)
print(b)
