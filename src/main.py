import os
import copy
import torch
import numpy as np
from tqdm import tqdm
from data import get_dataset
from torch.optim import AdamW
from predictors import MLP_Predictor
from model import GCN, GraphSAGE_GCN
from scheduler import CosineDecayScheduler
from BGRL import BGRL, compute_representations
from transforms import get_graph_drop_transform
from torch.nn.functional import cosine_similarity
from eval_function import fit_logistic_regression, fit_logistic_regression_preset_splits, fit_ppi_linear


import warnings
warnings.filterwarnings("ignore")


def train(step, model, optimizer, lr_scheduler, mm_scheduler, transform_1, transform_2, data):
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


def eval(model, dataset, device, args, train_masks=None, val_masks=None, test_masks=None):
    # make temporary copy of encoder
    tmp_encoder = copy.deepcopy(model.online_encoder).eval()
    representations, labels = compute_representations(tmp_encoder, dataset, device)
    val_scores = None

    if args.dataset == 'ppi':
        train_data = compute_representations(tmp_encoder, train_masks, device)
        val_data = compute_representations(tmp_encoder, val_masks, device)
        test_data = compute_representations(tmp_encoder, test_masks, device)
        num_classes = train_data[1].shape[1]
        val_scores, test_scores = fit_ppi_linear(num_classes, train_data, val_data, test_data, device,
                                                 args.num_eval_splits)
    elif args.dataset != 'wiki_cs':
        test_scores = fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy(),
                                              data_random_seed=args.data_seed, repeat=args.num_eval_splits)
    else:
        test_scores = fit_logistic_regression_preset_splits(representations.cpu().numpy(), labels.cpu().numpy(),
                                                            train_masks, val_masks, test_masks)

    return val_scores, test_scores


def main(args):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    dataset, train_masks, val_masks, test_masks = get_dataset(args.dataset_dir, args.dataset)

    data = dataset[0]
    data = data.to(device)

    # prepare transforms
    transform_1 = get_graph_drop_transform(drop_edge_p=args.drop_edge_p[0], feat_mask_p=args.feat_mask_p[0])
    transform_2 = get_graph_drop_transform(drop_edge_p=args.drop_edge_p[1], feat_mask_p=args.feat_mask_p[1])

    # build networks
    input_size, representation_size = data.ndata['feat'].size(1), args.graph_encoder_layer[-1]
    if args.dataset == 'ppi':
        encoder = GraphSAGE_GCN([input_size] + args.graph_encoder_layer)
    else:
        encoder = GCN([input_size] + args.graph_encoder_layer)
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=args.predictor_hidden_size)
    model = BGRL(encoder, predictor).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(args.lr, args.lr_warmup_epochs, args.epochs)
    mm_scheduler = CosineDecayScheduler(1 - args.mm, 0, args.epochs)

    val_scores, test_scores = [], []
    # train
    for epoch in tqdm(range(1, args.epochs + 1), desc='  - (Training)  '):
        # train(epoch - 1, model, optimizer, lr_scheduler, mm_scheduler, transform_1, transform_2, data)
        if epoch % args.eval_epochs == 0:
            val_scores, test_scores = eval(model, dataset, device, args, train_masks, val_masks, test_masks)
            if args.dataset == 'ppi':
                print('Epoch: {:04d} | Best Val F1: {:.4f} | Test F1: {:.4f}'.format(epoch, np.mean(val_scores),
                                                                                     np.mean(test_scores)))
            else:
                print('Epoch: {:04d} | Test Accuracy: {:.4f}'.format(epoch, np.mean(test_scores)))

    # save encoder weights
    if not os.path.isdir(args.weights_dir):
        os.mkdir(args.weights_dir)
    torch.save({'model': model.online_encoder.state_dict()}, os.path.join(args.weights_dir,
                                                                          'bgrl-{}.pt'.format(args.dataset)))

    if not os.path.isdir('../results'):
        os.mkdir('../results')
    with open('../results/{}.txt'.format(args.dataset), 'w') as f:
        f.write('{}, {}\n'.format(np.mean(test_scores), np.std(test_scores)))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Dataset options.
    parser.add_argument('--dataset', type=str, default='amazon_photos', choices=['coauthor_cs', 'coauthor_physics',
                                                                                 'amazon_photos', 'amazon_computers',
                                                                                 'wiki_cs', 'ppi'])
    parser.add_argument('--dataset_dir', type=str, default='../data')

    # Model options.
    parser.add_argument('--graph_encoder_layer', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--predictor_hidden_size', type=int, default=512)

    # Training options.
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--mm', type=float, default=0.99)
    parser.add_argument('--lr_warmup_epochs', type=int, default=1000)
    parser.add_argument('--weights_dir', type=str, default='../weights')

    # Augmentations options.
    parser.add_argument('--drop_edge_p', type=float, nargs='+', default=[0., 0.])
    parser.add_argument('--feat_mask_p', type=float, nargs='+', default=[0., 0.])

    # Evaluation options.
    parser.add_argument('--eval_epochs', type=int, default=250)
    parser.add_argument('--num_eval_splits', type=int, default=20)
    parser.add_argument('--data_seed', type=int, default=1)

    args = parser.parse_args()

    main(args)

