import torch.nn as nn
import torch
import torch.nn.functional as F


class DiscriminatorLoss(nn.Module):
    def __init__(self, net_d, device):
        super(DiscriminatorLoss, self).__init__()
        self.net_d = net_d
        self.device = device

    def forward(self, t, y):
        # ps = t.size(dim=2)
        #epsilon = torch.tensor([[10e-09]], requires_grad=False).to(device=self.device, dtype=torch.float32)
        loss_real = -torch.mean(torch.log(torch.sigmoid(self.net_d(t)) + 1e-9))
        loss_generated = -torch.mean(torch.log(1 - torch.sigmoid(self.net_d(y.detach())) + 1e-9))
        # disc_loss = loss_real + loss_generated

        return loss_real, loss_generated


def adversarial_loss(net_d, y, device):
    ps = y.size(dim=2)
    #epsilon = torch.tensor([[10e-09]], requires_grad=False).to(device=device, dtype=torch.float32)
    adv_loss = -torch.mean(torch.log(torch.sigmoid(net_d(y)) + 1e-9))
    # adv_loss = (12*ps*ps)*torch.mean(adv_loss)

    return adv_loss