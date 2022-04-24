import dgl
import json
import itertools
import numpy as np
import torch
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset, AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from transforms import NormalizeFeatures


def get_dataset(root, name, transform=NormalizeFeatures()):
    dgl_dataset_dict = {
        'coauthor_cs': CoauthorCSDataset,
        'coauthor_physics': CoauthorPhysicsDataset,
        'amazon_computers': AmazonCoBuyPhotoDataset,
        'amazon_photos': AmazonCoBuyComputerDataset,
    }

    dataset_class = dgl_dataset_dict[name]
    dataset = dataset_class(raw_dir=root, transform=transform)

    return dataset


def get_wiki_cs():
    data = json.load(open('../data/wiki_cs/data.json'))
    features = torch.FloatTensor(np.array(data['features']))
    labels = torch.LongTensor(np.array(data['labels']))
    train_masks = np.array(data['train_masks']).astype(bool).T
    val_masks = np.array(data['val_masks']).astype(bool).T
    test_mask = np.array(data['test_mask']).astype(bool)

    edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
    edges = list(itertools.chain(*edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    g = dgl.graph((edge_index[0], edge_index[1]))
    g = dgl.to_bidirected(g)
    g.ndata['feat'] = features
    g.ndata['label'] = labels
    g = NormalizeFeatures()(g)
    std, mean = torch.std_mean(g.ndata['feat'], dim=0, unbiased=False)
    g.ndata['feat'] = (g.ndata['feat'] - mean) / std

    return [g], train_masks, val_masks, test_mask
