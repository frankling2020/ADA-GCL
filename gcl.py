import torch
from torch.optim import Adam
import torch_geometric

from torch_geometric.loader import DataLoader
from torch_geometric import transforms
from torch_geometric.datasets import TUDataset
import torch_scatter

from utils import get_split, LREvaluator
from utils import feature_drop_weights, drop_feature_weighted, drop_edge_weighted

from tqdm import tqdm
from torch.optim import Adam


import matplotlib.pyplot as plt
import os.path as osp
import argparse

from model import ADA



def train(encoder_model, dataloader, optimizer, tau=0.2, alpha=1, reg=0, device='cpu'):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float, device=data.batch.device)
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        z, _ = encoder_model(x, edge_index, batch)

        edge_weights, node_feat_weights = encoder_model.edge_weight(z=z.clone().detach(), edge_index=edge_index)

        gs = []
        gs_ad = []
        for drop_rate in torch.tensor([.3, .4], dtype=torch.float):
            edge = edge_index.clone()
            edge_prob = feature_drop_weights(edge_weights, tau, device=device)
            # feat_prob = feature_drop_weights(node_feat_weights, tau, device=device)
            
            # x_aug = drop_feature_weighted(x, feat_prob, 0.1)
            edge = drop_edge_weighted(edge, edge_prob, drop_rate)
            
            _, g_aug = encoder_model.encoder(x, edge, batch)
            encoder_model.encoder_update()
            _, g_ad = encoder_model.augment_encoder(x, edge, batch)
            gs.append(g_aug)
            gs_ad.append(g_ad)

        g1, g2 = gs[0], gs[1]

        t1 = encoder_model.mlp(g1)
        t2 = encoder_model.mlp(g2)
        
        t3 = encoder_model.aug_mlp(gs_ad[0])
        t4 = encoder_model.aug_mlp(gs_ad[1])

        loss = encoder_model.loss(t1, t2) - alpha * (encoder_model.loss(t3, t4)) + reg * encoder_model.reg_loss()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss



def test(encoder_model, dataloader, device='cpu'):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float, device=data.batch.device)
        _, g = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    # result = SVMEvaluator(linear=True)(x, y, split)
    result = LREvaluator()(x, y, split)
    # result = RFEvaluator()(x, y, split)
    return result



def run(args):
    torch.cuda.empty_cache()
    datasets_name = args.dataset
    batch_size = args.batch_size
    device = torch.device(args.device)
    path = osp.join(osp.expanduser('.'), 'datasets')
    # transform = transforms.Compose([transforms.RemoveIsolatedNodes(), transforms.TargetIndegree()])
    dataset = TUDataset(path, name=datasets_name)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    input_dim = max(dataset.num_features, 1)


    encoder_model = ADA(input_dim, args.hidden_dims, args.num_layers, tau=args.tau).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Total Parameters: {encoder_model.total_parameters}")
    epochs = args.epochs
    alpha = args.alpha
    reg = args.reg
    losses = []
    test_result = test(encoder_model, dataloader, device)
    f = open(args.log, "a")
    f.write(f'{batch_size} batch-sized {datasets_name} before training test F1Mi={test_result["test"]:.3f}, valid F1Ma={test_result["valid"]:.3f}')
    with tqdm(total=epochs, desc='(T)') as pbar:
        for _ in range(1, 1+epochs):
            loss = train(encoder_model, dataloader, optimizer, args.tau, alpha, reg, device)
            losses.append(loss)
            pbar.set_postfix({'loss': loss})
            pbar.update()
    
    
    test_result = test(encoder_model, dataloader, device)
    print(f'(E): Best test F1Mi={test_result["test"]:.3f}, valid F1Mi={test_result["valid"]:.3f}')

    plt.title(f'{batch_size} batch-sized {datasets_name} with \n Best test F1Mi(accuracy)={test_result["test"]:.3f}, valid F1Mi={test_result["valid"]:.3f}')
    plt.plot(losses, label="Contrastive loss")
    plt.legend()
    plt.savefig(f"./figures/result_{datasets_name}.jpg", dpi=300)
    plt.close()
    print("Plot!")

    torch.save(encoder_model, f"./model_params/encoder_mdl_{datasets_name}.pth")
    print("Save!")

    f.write(f' {batch_size} batch-sized {datasets_name} with Best F1Mi(accuracy)={test_result["test"]:.3f}, valid F1Ma={test_result["valid"]:.3f} \n\n')
    f.close()


def arg_parser():
    parser = argparse.ArgumentParser(description='AD-AG-CL')
    parser.add_argument('--iter', type=int, default=5, help='Iteration')
    parser.add_argument('--dataset', type=str, default='MUTAG', help='Dataset')
    parser.add_argument('--batch_size', type=int, default=256, help="Batch Size")
    parser.add_argument('--hidden_dims', type=int, default=32, help="hidden dims")
    parser.add_argument('--num_layers', type=int, default=2, help="num of layers")
    parser.add_argument('--device', type=str, default='cuda', help='Device for training')
    parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="Weight decay for Adam")
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--alpha', type=float, default=1.0, help='adversarial attack weight')
    parser.add_argument('--tau', type=float, default=0.2, help='Temperature')
    parser.add_argument('--reg', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--log', type=str, default='log_test.txt', help='Record the training results')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    for _ in range(args.iter):
        run(args)
