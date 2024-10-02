import numpy as np
import torch


empty_tensor = torch.FloatTensor()
print(empty_tensor.shape)
x = torch.Tensor([1,2,3,4])
e = torch.argwhere(x == 0)
print(e.shape)
print(e)
print(torch.numel(e))