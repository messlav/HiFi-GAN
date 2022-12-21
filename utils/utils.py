import torch
import numpy as np
import random


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)
