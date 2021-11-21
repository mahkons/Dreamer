import torch
import torch.nn as nn
import torchvision.transforms as T

from .MADE import MADE
from .Shuffle import Shuffle
from .Flow import SequentialConditionalFlow
from .NormFunctions import ActNorm
from utils.logger import log

class MAF(nn.Module):
    def __init__(self, flow_dim, condition_dim, hidden_dim, num_blocks, device):
        super(MAF, self).__init__()
        self.flow_dim = flow_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.model = SequentialConditionalFlow(sum(
            [[MADE(flow_dim, condition_dim, hidden_dim), ActNorm(flow_dim), Shuffle(torch.randperm(flow_dim))] \
                for _ in range(num_blocks - 1)] \
            + [[MADE(flow_dim, condition_dim, hidden_dim)]], 
        []))
        self.model.to(device)
        self.prior = torch.distributions.Normal(torch.tensor(0., device=device),
                torch.tensor(1., device=device))

        self.initialized = False


    def calc_loss(self, inputs, conditions):
        inputs = inputs.reshape(-1, self.flow_dim)
        conditions = conditions.reshape(-1, self.condition_dim)

        if not self.initialized:
            with torch.no_grad():
                self.model.data_init(inputs, conditions)
            self.initialized = True
        loss = -self._get_log_prob(inputs, conditions).mean()

        return loss

    def forward_flow(self, inputs, conditions):
        in_shape = inputs.shape
        inputs = inputs.reshape(-1, self.flow_dim)
        conditions = conditions.reshape(-1, self.condition_dim)
        
        z, logjac = self.model.forward_flow(inputs, conditions)
        return z.reshape(in_shape), logjac

    def get_log_prob(self, inputs, conditions):
        return self._get_log_prob(inputs, conditions)

    def _get_log_prob(self, inputs, conditions):
        prediction, log_jac = self.model.forward_flow(inputs, conditions)
        log_prob = self.prior.log_prob(prediction).sum(dim=1) + log_jac
        return log_prob

    def sample(self, conditions):
        batch_size = conditions.shape[0]
        with torch.no_grad():
            z = self.prior.sample([batch_size, self.flow_dim])
            x, _ = self.model.inverse_flow(z, conditions)
        return x


    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])


"""
    uniform noise to dequantize input
    logit(a + (1 - 2a) * image) as in paper
"""
class MAFImageTransform():
    def __init__(self, dataset):
        if dataset == "mnist":
            self.base_transform = T.Compose([T.ToTensor(), T.RandomHorizontalFlip()])
            self.alpha = 0.01
        else:
            raise AttributeError("Unknown dataset")


    def __call__(self, image):
        image = self.base_transform(image)
        noise = (torch.rand_like(image) - 0.5) * (1/256.)
        image = (image + noise).clip(0., 1.)
        return torch.logit(self.alpha +  (1 - 2 * self.alpha) * image)

