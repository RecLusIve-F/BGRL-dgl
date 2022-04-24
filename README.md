# DGL Implementations of BGRL

This DGL example implements the GNN model proposed in the paper [Large-Scale Representation Learning on Graphs via Bootstrapping](https://arxiv.org/abs/2102.06514). For the original implementation, see [here](https://github.com/nerdslab/bgrl).

Contributor: [RecLusIve-F](https://github.com/RecLusIve-F)

### Requirements

The codebase is implemented in Python 3.8. For version requirement of packages, see below.

```
dgl 0.8.1
numpy 1.21.2
torch 1.10.2
scikit-learn 1.0.2
```

### Usage

###### Dataset options


###### GPU options


###### Model options


###### Examples
```bash
# Transductive learning
# coauthor_cs
python main.py --dataset coauthor_cs --graph_encoder_layer 512 256 --drop_edge_p 0.3 0.2 --feat_mask_p 0.3 0.4

# coauthor_physics
python main.py --dataset coauthor_physics --graph_encoder_layer 256 128 --drop_edge_p 0.4 0.1 --feat_mask_p 0.1 0.4

# wiki_cs
python main.py --dataset wiki_cs --graph_encoder_layer 512 256 --drop_edge_p 0.2 0.3 --feat_mask_p 0.2 0.1 --lr 5e-4

# amazon_photos
python main.py --dataset amazon_photos --graph_encoder_layer 256 128 --drop_edge_p 0.4 0.1 --feat_mask_p 0.1 0.2 --lr 1e-4

# amazon_computers
python main.py --dataset amazon_computers --graph_encoder_layer 256 128 --drop_edge_p 0.5 0.4 --feat_mask_p 0.2 0.1 --lr 5e-4
```

### Performance




