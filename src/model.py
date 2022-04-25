import torch
from torch import nn
from torch.nn import BatchNorm1d, Parameter
from torch.nn.init import ones_, zeros_
from dgl.nn.pytorch.conv import GraphConv, SAGEConv


class LayerNorm(nn.Module):
    def __init__(self, in_channels, eps=1e-5, affine=True):
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps

        if affine:
            self.weight = Parameter(torch.Tensor(in_channels))
            self.bias = Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        ones_(self.weight)
        zeros_(self.bias)

    def forward(self, x, batch=None):
        device = x.device
        if batch is None:
            x = x - x.mean()
            out = x / (x.std(unbiased=False) + self.eps)
        else:
            batch_size = int(batch.max()) + 1
            batch_idx = [batch == i for i in range(batch_size)]
            norm = torch.tensor([i.sum() for i in batch_idx], dtype=x.dtype).clamp_(min=1).to(device)
            norm = norm.mul_(x.size(-1)).view(-1, 1)
            tmp_list = [x[i] for i in batch_idx]
            mean = torch.concat([i.sum(0).unsqueeze(0) for i in tmp_list], dim=0).sum(dim=-1, keepdim=True).to(device)
            mean = mean / norm
            x = x - mean.index_select(0, batch.long())
            var = torch.concat([(i * i).sum(0).unsqueeze(0) for i in tmp_list], dim=0).sum(dim=-1, keepdim=True).to(device)
            var = var / norm
            out = x / (var + self.eps).sqrt().index_select(0, batch.long())

        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'


class GCN(nn.Module):
    def __init__(self, layer_sizes, batch_norm_mm=0.99):
        super(GCN, self).__init__()

        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(GraphConv(in_dim, out_dim, allow_zero_in_degree=True))
            self.layers.append(BatchNorm1d(out_dim, momentum=batch_norm_mm))
            self.layers.append(nn.PReLU())

    def forward(self, g):
        x = g.ndata['feat']
        for layer in self.layers:
            if isinstance(layer, GraphConv):
                x = layer(g, x)
            else:
                x = layer(x)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class GraphSAGE_GCN(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()

        input_size, hidden_size, embedding_size = layer_sizes

        self.convs = nn.ModuleList([
            SAGEConv(input_size, hidden_size, 'mean'),
            SAGEConv(hidden_size, hidden_size, 'mean'),
            SAGEConv(hidden_size, embedding_size, 'mean')
        ])

        self.skip_lins = nn.ModuleList([
            nn.Linear(input_size, hidden_size, bias=False),
            nn.Linear(input_size, hidden_size, bias=False),
        ])

        self.layer_norms = nn.ModuleList([
            LayerNorm(hidden_size),
            LayerNorm(hidden_size),
            LayerNorm(embedding_size),
        ])

        self.activations = nn.ModuleList([
            nn.PReLU(),
            nn.PReLU(),
            nn.PReLU(),
        ])

    def forward(self, g):
        x = g.ndata['feat']
        if 'batch' in g.ndata.keys():
            batch = g.ndata['batch']
        else:
            batch = None

        h1 = self.convs[0](g, x)
        h1 = self.layer_norms[0](h1, batch)
        h1 = self.activations[0](h1)

        x_skip_1 = self.skip_lins[0](x)
        h2 = self.convs[1](g, h1 + x_skip_1)
        h2 = self.layer_norms[1](h2, batch)
        h2 = self.activations[1](h2)

        x_skip_2 = self.skip_lins[1](x)
        ret = self.convs[2](g, h1 + h2 + x_skip_2)
        ret = self.layer_norms[2](ret, batch)
        ret = self.activations[2](ret)
        return ret

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        for m in self.skip_lins:
            m.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)
        for m in self.layer_norms:
            m.reset_parameters()
