import torch
import torch.nn as nn

from models.MLP import MLP


class RewardNetwork(MLP):
    def __init__(self, state_dim):
        super(RewardNetwork, self).__init__(state_dim, 1, [300, 300], nn.GELU)

    def forward(self, x):
        x = super().forward(x)
        return x.squeeze(len(x.shape) - 1)



class DiscountNetwork(MLP):
    def __init__(self, state_dim):
        super(DiscountNetwork, self).__init__(state_dim, 1, [300, 300], nn.GELU)

    def forward(self, x):
        assert(False)

    def predict_logit(self, x):
        x = super().forward(x)
        return x.squeeze(len(x.shape) - 1)


""" flattens several last dims """
class MyFlatten(nn.Module):
    def __init__(self, n_last_dims):
        super(MyFlatten, self).__init__()
        self.n = n_last_dims

    def forward(self, x):
        return x.view(list(x.shape)[:-self.n] + [-1])


class ObservationEncoder(nn.Module):
    def __init__(self, obs_dim, out_dim, from_pixels):
        super(ObservationEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.out_dim = out_dim
        self.from_pixels = from_pixels
        
        self.model = self._build_model()

    def forward(self, x):
        if len(x.shape) > 4 and self.from_pixels: # convolutional layers does not accept 5d tensors
            out_shape = list(x.shape)[:-3] + [self.out_dim]
            return self.model(x.reshape(-1, *self.obs_dim)).reshape(out_shape)
        return self.model(x)

    def _build_model(self):
        if not self.from_pixels:
            return MLP(self.obs_dim, self.out_dim, [300, 300], nn.GELU)

        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            MyFlatten(3),
            nn.ReLU(),
            nn.Linear(1024, self.out_dim)
        )


class MyUnflattenModule(nn.Module):
    def __init__(self, n_new_dims):
        super(MyUnflattenModule, self).__init__()
        self.n = n_new_dims

    def forward(self, x):
        return x.view(list(x.shape) + [1] * self.n)

class ObservationDecoder(nn.Module):
    def __init__(self, in_dim, obs_dim, from_pixels):
        super(ObservationDecoder, self).__init__()
        self.in_dim = in_dim
        self.obs_dim = obs_dim
        self.from_pixels = from_pixels
        
        self.model = self._build_model()

    def forward(self, x):
        if len(x.shape) > 2 and self.from_pixels: # convolutional layers does not accept 5d tensors
            out_shape = list(x.shape)[:-1] + [*self.obs_dim]
            return self.model(x.reshape(-1, self.in_dim)).reshape(out_shape)
        return self.model(x)

    def _build_model(self):
        if not self.from_pixels:
            return MLP(self.in_dim, self.obs_dim, [300, 300], nn.GELU)

        return nn.Sequential(
            nn.Linear(self.in_dim, 1024),
            MyUnflattenModule(2),
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)
        )

