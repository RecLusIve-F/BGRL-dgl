import time
import numpy as np
from torch import nn
from torch.nn import BatchNorm1d, LayerNorm
from dgl.nn.pytorch.conv import GraphConv
from tqdm import tqdm


class GCN(nn.Module):
    def __init__(self, layer_sizes, batch_norm=False, batch_norm_mm=0.99):
        super(GCN, self).__init__()
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]

        self.layers = list()
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(GraphConv(in_dim, out_dim, allow_zero_in_degree=True))
            if batch_norm:
                self.layers.append(BatchNorm1d(out_dim, momentum=batch_norm_mm))
            else:
                self.layers.append(LayerNorm(out_dim))

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


if __name__ == '__main__':
    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 101):
            loss = np.random.randint(0, 100)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            time.sleep(1)

