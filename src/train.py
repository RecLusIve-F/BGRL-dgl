import copy
import torch
import numpy as np
from tqdm import tqdm
from model import GCN
from torch.optim import AdamW
from predictors import MLP_Predictor
from data import get_dataset, get_wiki_cs
from scheduler import CosineDecayScheduler
from BGRL import BGRL, compute_representations
from transforms import get_graph_drop_transform
from torch.nn.functional import cosine_similarity
from logistic_regression_eval import fit_logistic_regression, fit_logistic_regression_preset_splits

import warnings
warnings.filterwarnings("ignore")


def train(step, model, optimizer, lr_scheduler, mm_scheduler, transform_1, transform_2, data, device):
    model.train()

    # update learning rate
    lr = lr_scheduler.get(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # update momentum
    mm = 1 - mm_scheduler.get(step)

    # forward
    optimizer.zero_grad()

    x1, x2 = transform_1(data), transform_2(data)

    q1, y2 = model(x1, x2)
    q2, y1 = model(x2, x1)

    loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
    loss.backward()

    # update online network
    optimizer.step()
    # update target network
    model.update_target_network(mm)

    return loss.item()


def eval(epoch, model, dataset, device, args, train_masks=None, val_masks=None, test_masks=None):
    # make temporary copy of encoder
    tmp_encoder = copy.deepcopy(model.online_encoder).eval()
    representations, labels = compute_representations(tmp_encoder, dataset, device)

    if args.dataset != 'wiki_cs':
        scores = fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy(),
                                         data_random_seed=args.data_seed, repeat=args.num_eval_splits)
    else:
        scores = fit_logistic_regression_preset_splits(representations.cpu().numpy(), labels.cpu().numpy(),
                                                       train_masks, val_masks, test_masks)

    return np.mean(scores)


def main(args):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    train_masks, val_masks, test_masks = None, None, None

    # load data
    if args.dataset != 'wiki_cs':
        dataset = get_dataset(args.dataset_dir, args.dataset)
    else:
        dataset, train_masks, val_masks, test_masks = get_wiki_cs()

    data = dataset[0]  # all data include one graph
    data = data.to(device)
    data.ndata['feat'] = data.ndata['feat'].to(device)

    # prepare transforms
    transform_1 = get_graph_drop_transform(drop_edge_p=args.drop_edge_p_1, feat_mask_p=args.feat_mask_p_1)
    transform_2 = get_graph_drop_transform(drop_edge_p=args.drop_edge_p_2, feat_mask_p=args.feat_mask_p_2)

    # build networks
    input_size, representation_size = data.ndata['feat'].size(1), args.graph_encoder_layer[-1]
    encoder = GCN([input_size] + args.graph_encoder_layer, device, batch_norm=True)
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=args.predictor_hidden_size)
    model = BGRL(encoder, predictor).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(args.lr, args.lr_warmup_epochs, args.epochs)
    mm_scheduler = CosineDecayScheduler(1 - args.mm, 0, args.epochs)

    for epoch in tqdm(range(1, args.epochs + 1), desc='  - (Training)  '):
        train_loss = train(epoch - 1, model, optimizer, lr_scheduler, mm_scheduler, transform_1, transform_2, data,
                           device)
        if epoch % args.eval_epochs == 0:
            val_score = eval(epoch, model, dataset, device, args, train_masks, val_masks, test_masks)
            tqdm.write('Epoch: {:04d} | Train Loss: {:.4f} | Val Score: {:.4f}'.format(epoch, train_loss, val_score))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('--data_seed', type=int, default=1)
    parser.add_argument('--num_eval_splits', type=int, default=3)

    # Dataset.
    parser.add_argument('--dataset_dir', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='amazon_photos', choices=['coauthor_cs', 'coauthor_physics',
                                                                                 'amazon_photos', 'amazon_computers',
                                                                                 'wiki_cs', 'ppi'])
    # Architecture.
    parser.add_argument('--graph_encoder_layer', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--predictor_hidden_size', type=int, default=512)

    # Training hyper-parameters.
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--mm', type=float, default=0.99)

    parser.add_argument('--lr_warmup_epochs', type=int, default=1000)

    # Augmentations.
    parser.add_argument('--drop_edge_p_1', type=float, default=0.1)
    parser.add_argument('--drop_edge_p_2', type=float, default=0.1)
    parser.add_argument('--feat_mask_p_1', type=float, default=0.1)
    parser.add_argument('--feat_mask_p_2', type=float, default=0.1)

    # Evaluation
    parser.add_argument('--eval_epochs', type=int, default=5)
    args = parser.parse_args()

    main(args)

