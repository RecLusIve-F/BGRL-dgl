import dgl
import copy
import torch
from torch import nn


class BGRL(nn.Module):
    r"""BGRL architecture for Graph representation learning.
    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.
    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, encoder, predictor):
        super(BGRL, self).__init__()
        # online network
        self.online_encoder = encoder
        self.predictor = predictor

        # target network
        self.target_encoder = copy.deepcopy(encoder)

        # reinitialize weights
        self.target_encoder.reset_parameters()

        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.
        Args:
            mm (float): Momentum used in moving average update.
        """
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, online_x, target_x):
        # forward online network
        online_y = self.online_encoder(online_x)

        # prediction
        online_q = self.predictor(online_y)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x).detach()
        return online_q, target_y


def compute_representations(net, dataset, device):
    r"""Pre-computes the representations for the entire data.
    Returns:
        [torch.Tensor, torch.Tensor]: Representations and labels.
    """
    net.eval()
    reps = []
    labels = []

    if len(dataset) == 1:
        data = dataset[0]
        data = dgl.add_self_loop(data)
        data = data.to(device)
        with torch.no_grad():
            reps.append(net(data))
            labels.append(data.ndata['label'])
    else:
        for data in dataset:
            # forward
            data = data.to(device)
            with torch.no_grad():
                reps.append(net(data))
                labels.append(data.ndata['label'])

    reps = torch.cat(reps, dim=0)
    labels = torch.cat(labels, dim=0)
    return [reps, labels]
