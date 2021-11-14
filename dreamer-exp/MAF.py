import torch
import torch.nn as nn
import torchvision.transforms as T

from models import MADE, Reverse, SequentialFlow, ActNorm

MAX_GRAD_NORM = 100.
WEIGHT_DECAY = 1e-6

class MAF():
    def __init__(self, image_shape, condition_dim, hidden_dim, num_blocks, lr, device):
        assert(len(image_shape) == 3)
        self.image_shape = image_shape
        self.condition_dim = condition_dim
        self.D = torch.prod(torch.tensor(image_shape))
        self.device = device

        self.model = SequentialFlow(sum(
            [[MADE(self.D, condition_dim, hidden_dim), ActNorm(self.D), Reverse()] for _ in range(num_blocks - 1)] \
            + [[MADE(self.D, condition_dim, hidden_dim)]], 
        []))
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=1000)
        self.prior = torch.distributions.Normal(torch.tensor(0., device=device),
                torch.tensor(1., device=device))

        self.initialized = False


    def train(self, images, conditions):
        images = torch.flatten(images.to(self.device), start_dim=1)
        if not self.initialized:
            with torch.no_grad():
                self.model.data_init(images, conditions)
            self.initialized = True
        loss = -self._get_log_prob(images, conditions).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def encode(self, images, conditions):
        images = torch.flatten(images.to(self.device), start_dim=1)
        prediction, log_jac = self.model.forward_flow(images, conditions)[0]
        return prediction

    def get_log_prob(self, images, conditions):
        images = torch.flatten(images.to(self.device), start_dim=1)
        return self._get_log_prob(images)

    def _get_log_prob(self, images, conditions):
        prediction, log_jac = self.model.forward_flow(images, conditions)
        log_prob = self.prior.log_prob(prediction).sum(dim=1) + log_jac
        return log_prob

    def sample(self, batch_size):
        with torch.no_grad():
            z = self.prior.sample([batch_size, self.D])
            x, _ = self.model.inverse_flow(z)
            x = x.reshape([batch_size] + list(self.image_shape))
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




