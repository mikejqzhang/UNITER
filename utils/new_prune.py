import torch
import torch.nn as nn

class Pruner():
    def __init__(self, parameters):
        self.parameters = parameters
        for module, name in parameters:
            setattr(module,
                    name + '_mask',
                    torch.ones_like(getattr(module, name))
                    )

    def prune(self, sparsity):

        t = torch.nn.utils.parameters_to_vector(
                [getattr(module, name) for module, name in self.parameters])
        mask = torch.nn.utils.parameters_to_vector(
                [getattr(module, name + "_mask")
                    for (module, name) in self.parameters]).clone()

        n_params = t.numel()
        n_params_to_prune = int(n_params * sparsity)

        topk = torch.topk(torch.abs(t * mask).view(-1),
                k=n_params_to_prune, largest=False)
        mask.view(-1)[topk.indices] = 0

        pointer = 0
        for module, name in self.parameters:
            param = getattr(module, name)
            param_shape = param.size()
            num_param = param.numel()
            param_mask = mask[pointer : pointer + num_param].view_as(param)
            new_param = param_mask * param
            del module._parameters[name]
            module.register_parameter(name, nn.Parameter(new_param))
            pointer += num_param
