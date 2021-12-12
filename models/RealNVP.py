import torch
import torch.nn as nn
import torchvision.transforms as T

from .Flow import SequentialConditionalFlow
from .CouplingLayer import CouplingLayer
from .NormFunctions import ActNorm

SCALE_L2_REG_COEFF = 5e-5
MAX_GRAD_NORM = 100.

class RealNVP(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_shape, num_coupling, num_hidden, device):
        super(RealNVP, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.device = device

        mask = torch.arange(input_dim) % 2
        modules = []
        for i in range(num_coupling):
            modules.append(ActNorm(input_dim))
            modules.append(CouplingLayer(input_dim, condition_dim, hidden_shape, num_hidden,
                mask if i % 2 == 0 else 1 - mask))
        self.model = SequentialConditionalFlow(modules)
        self.model.to(device)

        self.prior = torch.distributions.Normal(torch.tensor(0., device=device),
                torch.tensor(1., device=device))

        self.initialized = True # no data init


    def forward_flow(self, inputs, conditions):
        in_shape = inputs.shape
        inputs = inputs.reshape(-1, self.input_dim)
        conditions.reshape(inputs.shape[0], self.condition_dim)

        if not self.initialized and inputs.shape[0] != 1: # hack todo fix?
            with torch.no_grad():
                self.model.data_init(inputs, conditions)
            self.initialized = True
        
        z, logjac = self.model.forward_flow(inputs, conditions)
        return z.reshape(in_shape), logjac.reshape(in_shape[:-1])

    def inverse_flow(self, z, conditions):
        in_shape = inputs.shape
        inputs = inputs.reshape(-1, self.input_dim)
        conditions.reshape(inputs.shape[0], self.condition_dim)

        x, logjac = self.model.inverse_flow(z, conditions)
        return x.reshape(in_shape), logjac.reshape(in_shape[:-1])

    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])


"""
    for celeba center crop and resize as in paper
    uniform noise to dequantize input
    logit(a + (1 - 2a) * image) as in paper
"""
class RealNVPImageTransform():
    def __init__(self, dataset):
        if dataset == "celeba":
            self.base_transform = T.Compose([T.ToTensor(), T.CenterCrop((148, 148)), T.Resize((64, 64)), T.RandomHorizontalFlip()])
            self.alpha = 0.05
        elif dataset == "mnist":
            self.base_transform = T.Compose([T.ToTensor(), T.RandomHorizontalFlip()])
            self.alpha = 0.01
        else:
            raise AttributeError("Unknown dataset")


    def __call__(self, image):
        image = self.base_transform(image)
        noise = (torch.rand_like(image) - 0.5) * (1/256.)
        image = (image + noise).clip(0., 1.)
        return torch.logit(self.alpha +  (1 - 2 * self.alpha) * image)




