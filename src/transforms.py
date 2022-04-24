import copy
import torch
from dgl.transforms import BaseTransform, Compose, DropEdge
from torch.distributions import Bernoulli


class NormalizeFeatures(BaseTransform):
    def __init__(self, node_feat_names=None):
        if node_feat_names is None:
            node_feat_names = ['feat']
        self.node_feat_names = node_feat_names

    def __call__(self, g):
        if self.node_feat_names:
            for feat_name in self.node_feat_names:
                feat = g.ndata[feat_name] - g.ndata[feat_name].min()
                feat.div_(feat.sum(dim=-1, keepdim=True).clamp_(min=1.))
                g.ndata[feat_name] = feat
        return g


class NodeFeaturesMasking(BaseTransform):
    def __init__(self, p=0.5, feat_names=None):
        if feat_names is None:
            feat_names = ['feat']
        self.p = p
        self.dist = Bernoulli(p)
        self.feat_names = feat_names

    def __call__(self, g):
        # Fast path
        if self.p == 0:
            return g

        for feat in self.feat_names:
            feat_mask = self.dist.sample(torch.Size([g.ndata[feat].shape[-1], ])).bool()
            g.ndata[feat][:, feat_mask] = 0
        return g


def get_graph_drop_transform(drop_edge_p, feat_mask_p):
    transforms = list()

    # make copy of graph
    transforms.append(copy.deepcopy)

    # drop edges
    if drop_edge_p > 0.:
        transforms.append(DropEdge(drop_edge_p))

    # drop features
    if feat_mask_p > 0.:
        transforms.append(NodeFeaturesMasking(feat_mask_p))
    return Compose(transforms)
