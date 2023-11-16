import torch
import numpy as np

class FitnessEstimator:
    def __init__(self, model, batchsize):
        self.batchsize = batchsize
        model.K = np.zeros((batchsize, batchsize))
        model.num_actfun = 0
        self.model = model


    def __call__(self,module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        out = out.view(out.size(0), -1)
        batch_num , neuron_num = out.size()
        x = (out > 0).float()

        full_matrix = torch.ones((self.batchsize, self.batchsize)).cuda() * neuron_num
        sparsity = (x.sum(1)/neuron_num).unsqueeze(1)
        norm_K = ((sparsity @ (1-sparsity.t())) + ((1-sparsity) @ sparsity.t())) * neuron_num
        rescale_factor = torch.div(0.5* torch.ones((self.batchsize, self.batchsize)).cuda(), norm_K+1e-3)
        K1_0 = (x @ (1 - x.t()))
        K0_1 = ((1-x) @ x.t())
        K_total = (full_matrix - rescale_factor * (K0_1 + K1_0))

        self.model.K = self.model.K + (K_total.cpu().numpy())
        self.model.num_actfun += 1