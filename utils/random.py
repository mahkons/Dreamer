import torch
import numpy as np
import random

def init_random_seeds(seed, cuda_determenistic):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if cuda_determenistic:
        torch.backends.cudnn.deterministic = cuda_determenistic
        torch.backends.cudnn.benchmark = cuda_determenistic
