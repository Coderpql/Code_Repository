import numpy as np
import torch

a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a.sum())

b = torch.tensor([[1, 2, 3],
                [4, 5, 6]])
print(b.sum(dim=1))