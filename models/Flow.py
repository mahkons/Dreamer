import torch
import torch.nn as nn

class ConditionalFlow(nn.Module):
    def __init__(self):
        super(ConditionalFlow, self).__init__()
        self.device = torch.device("cpu")

    def to(self, device):
        super(ConditionalFlow, self).to(device)
        self.device = device

    def forward_flow(self, x, condition):
        """ returns forward flow computation result and log of jacobian """
        raise NotImplementedError

    def inverse_flow(self, x, condition):
        """ returns inverse flow computation result and log of jacobian  """
        raise NotImplementedError

    def data_init(self, x, condition):
        return self.forward_flow(x, condition)[0]



class SequentialConditionalFlow(ConditionalFlow):
    def __init__(self, modules):
        super(SequentialConditionalFlow, self).__init__()

        self.flow_modules = list(modules)
        for idx, module in enumerate(self.flow_modules):
            assert(isinstance(module, ConditionalFlow))
            self.add_module(str(idx), module)

    def forward_flow(self, x, condition):
        log_sum = torch.zeros(x.shape[0], dtype=torch.float, device=self.device)
        for module in self.flow_modules:
            x, log_jac = module.forward_flow(x, condition)
            log_sum += log_jac
        return x, log_sum

    def inverse_flow(self, x, condition):
        log_sum = torch.zeros(x.shape[0], dtype=torch.float, device=self.device)
        for module in reversed(self.flow_modules):
            x, log_jac = module.inverse_flow(x, condition)
            log_sum += log_jac
        return x, log_sum

    def data_init(self, x, condition):
        for module in self.flow_modules:
            x = module.data_init(x, condition)
        return x
