import torch
import numpy as np

from params.constants import K, NU

class FSVDDLoss:

    def __init__(self):
        self.cuda_state = False

    def cuda(self):
        self.cuda_state = True

    def get_info(self):
        infos = f"K: {K}\n"
        infos += f"Nu: {NU}\n"

    def init_vars(self, model_mapping, model_flow, dataloader_train, eps=10e-3):

        print('Computing C...')

        nb_imgs = 0
        C = torch.zeros(model_flow.input_dim)
        if self.cuda_state:
            C = C.cuda()

        with torch.no_grad():
            for data, _, _ in dataloader_train:

                nb_imgs += data.shape[0]
                if self.cuda_state:
                    data = data.cuda()

                mapping = model_mapping.forward(data, False)
                output = model_flow.forward(mapping)
                C += torch.sum(output, dim=0)

        C /= nb_imgs
        C[(abs(C) < eps) & (C < 0)] = -eps
        C[(abs(C) < eps) & (C > 0)] = eps

        self.C = C
        self.R = 0.
        self.W = 0.

    def update_R(self, model, epoch, output):
        if epoch % K == 0:
            dist = torch.sum((output * (self.W ** (1 / model.input_dim)) - self.C) ** 2, dim=1)
            self.R = np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - NU)

    def update_W(self, model):
        self.W = torch.sum(model.scaling_diag)

    def __call__(self, model, output):
        """
        Return the loss given an output
        """
        dist = torch.sum((output * (self.W ** (1 / model.input_dim)) - self.C) ** 2, dim=1)
        scores = dist - self.R ** 2
        loss = self.R ** 2 + (1 / NU) * torch.mean(torch.max(torch.zeros_like(scores), scores))

        return loss
