import os
import copy
import json
import torch
import itertools
import numpy as np
from dgl.convert import graph
from dgl.dataloading import GraphDataLoader
from dgl.transforms import to_bidirected, reorder_graph
from dgl.transforms import Compose, DropEdge, FeatMask, RowFeatNormalizer
from dgl.data.utils import generate_mask_tensor, load_graphs, save_graphs, _get_dgl_url
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset, AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, PPIDataset, DGLBuiltinDataset


class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))


class WikiCSDataset(DGLBuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        _url = _get_dgl_url('dataset/wiki_cs.zip')
        super(WikiCSDataset, self).__init__(name='wiki_cs',
                                            raw_dir=raw_dir,
                                            url=_url,
                                            force_reload=force_reload,
                                            verbose=verbose,
                                            transform=transform)

    def process(self):
        """process raw data to graph, labels and masks"""
        with open(os.path.join(self.raw_path, 'data.json')) as f:
            data = json.load(f)
        features = torch.tensor(np.array(data['features']), dtype=torch.float32)
        labels = torch.tensor(np.array(data['labels']), dtype=torch.int64)

        train_masks = np.array(data['train_masks'], dtype=bool).T
        val_masks = np.array(data['val_masks'], dtype=bool).T
        stopping_masks = np.array(data['stopping_masks'], dtype=bool).T
        test_mask = np.array(data['test_mask'], dtype=bool)

        edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
        edges = np.array(list(itertools.chain(*edges)))
        src, dst = edges[:, 0], edges[:, 1]

        g = graph((src, dst))
        g = to_bidirected(g)

        g.ndata['feat'] = features
        g.ndata['label'] = labels
        g.ndata['train_mask'] = generate_mask_tensor(train_masks)
        g.ndata['val_mask'] = generate_mask_tensor(val_masks)
        g.ndata['stopping_mask'] = generate_mask_tensor(stopping_masks)
        g.ndata['test_mask'] = generate_mask_tensor(test_mask)

        g = reorder_graph(g, node_permute_algo='rcmk', edge_permute_algo='dst', store_ids=False)

        self._graph = g

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self._graph)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        g, _ = load_graphs(graph_path)
        self._graph = g[0]

    @property
    def num_classes(self):
        return 10

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        if self._transform is None:
            return self._graph
        else:
            return self._transform(self._graph)


def get_graph_drop_transform(drop_edge_p, feat_mask_p):
    transforms = list()

    # make copy of graph
    transforms.append(copy.deepcopy)

    # drop edges
    if drop_edge_p > 0.:
        transforms.append(DropEdge(drop_edge_p))

    # drop features
    if feat_mask_p > 0.:
        transforms.append(FeatMask(feat_mask_p, node_feat_names=['feat']))

    return Compose(transforms)


def get_wiki_cs(root, transform=RowFeatNormalizer()):
    dataset = WikiCSDataset(root, transform=transform)
    g = dataset[0]
    std, mean = torch.std_mean(g.ndata['feat'], dim=0, unbiased=False)
    g.ndata['feat'] = (g.ndata['feat'] - mean) / std

    return [g]


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


def get_dataset(root, name, transform=RowFeatNormalizer()):
    dgl_dataset_dict = {
        'coauthor_cs': CoauthorCSDataset,
        'coauthor_physics': CoauthorPhysicsDataset,
        'amazon_computers': AmazonCoBuyComputerDataset,
        'amazon_photos': AmazonCoBuyPhotoDataset,
        'wiki_cs': get_wiki_cs,
        'ppi': get_ppi
    }

    dataset_class = dgl_dataset_dict[name]
    train_data, val_data, test_data = None, None, None
    if name != 'ppi':
        dataset = dataset_class(root, transform=transform)
    else:
        dataset, train_data, val_data, test_data = dataset_class(root, transform=transform)

    return dataset, train_data, val_data, test_data