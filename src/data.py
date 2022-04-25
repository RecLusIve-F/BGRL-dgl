import dgl
import json
import torch
import itertools
import numpy as np
from transforms import NormalizeFeatures
from dgl.dataloading import GraphDataLoader
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset, AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, PPIDataset


def get_dataset(root, name, transform=NormalizeFeatures()):
    dgl_dataset_dict = {
        'coauthor_cs': CoauthorCSDataset,
        'coauthor_physics': CoauthorPhysicsDataset,
        'amazon_computers': AmazonCoBuyComputerDataset,
        'amazon_photos': AmazonCoBuyPhotoDataset,
        'wiki_cs': get_wiki_cs,
        'ppi': get_ppi
    }

    dataset_class = dgl_dataset_dict[name]

    return dataset_class(root, transform=transform)


def get_wiki_cs(root, transform=NormalizeFeatures()):
    data = json.load(open('{}/wiki_cs/data.json'.format(root)))
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
    g = transform(g)
    std, mean = torch.std_mean(g.ndata['feat'], dim=0, unbiased=False)
    g.ndata['feat'] = (g.ndata['feat'] - mean) / std

    return [g], train_masks, val_masks, test_mask


def get_ppi(root, transform=None):
    train_dataset = PPIDataset(mode='train', raw_dir=root)
    val_dataset = PPIDataset(mode='valid', raw_dir=root)
    test_dataset = PPIDataset(mode='test', raw_dir=root)
    train_val_dataset = [i for i in train_dataset] + [i for i in val_dataset]
    for idx, data in enumerate(train_val_dataset):
        data.ndata['batch'] = torch.zeros(data.number_of_nodes()) + idx
        data.ndata['batch'] = data.ndata['batch'].long()

    g = list(GraphDataLoader(train_val_dataset, batch_size=22, shuffle=True))

    return g, PPIDataset(mode='train', raw_dir=root), PPIDataset(mode='valid', raw_dir=root), test_dataset

