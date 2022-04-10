import os.path as osp
from pickletools import optimize
from tqdm import tqdm

import csv
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam, RAdam

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from encoder import ADA
from arg_parser import arg_parser
from train import train
from test import test


def run(args):
    datasets_name = args.dataset
    batch_size = args.batch_size
    device = torch.device(args.device)
    path = osp.join(osp.expanduser('.'), 'datasets')


    ##################################################################################################################################
    # Setup
    ##################################################################################################################################
    dataset = TUDataset(path, name=datasets_name)

    dataloader = DataLoader(dataset, batch_size=batch_size)
    input_dim = max(dataset.num_features, 1)

    encoder_model = ADA(input_dim, args.hidden_dims, args.num_layers, edge_dims=args.edge_dims).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = RAdam(encoder_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)



    ##################################################################################################################################
    # Print the information
    ##################################################################################################################################
    print(f"Trainable Parameters: {encoder_model.total_parameters}")
    epochs = args.epochs

    

    
    ##################################################################################################################################
    # first time test block 
    ##################################################################################################################################
    losses = []
    test_result = test(encoder_model, dataloader, args)

    before_test = test_result['test']
    before_valid = test_result['valid']
    # before_test = test_result['micro_f1']
    
    f = open(args.log, "a")
    
    f.write(f'{batch_size} batch-sized {datasets_name} before training test F1Mi={test_result["test"]:.3f}, valid F1Mi={test_result["valid"]:.3f}')
    # f.write(f'{batch_size} batch-sized {datasets_name} before training test F1Mi={before_test:.3f}')

    
    
    ##################################################################################################################################
    # Train
    ##################################################################################################################################
    with tqdm(total=epochs, desc='(T)') as pbar:
        for _ in range(1, 1+epochs):
            loss, _ = train(encoder_model, dataloader, optimizer, args)
            losses.append(loss)
            pbar.set_postfix({'loss': loss})
            pbar.update()
    
    
    
    ##################################################################################################################################
    # Valid
    ##################################################################################################################################
    test_result = test(encoder_model, dataloader, args)
    print(f'(E): Best test F1Mi={test_result["test"]:.3f}, train F1Mi={test_result["train"]:.3f}, valid F1Mi={test_result["valid"]:.3f}')
    # print(f'(E): Best test F1Mi={test_result["micro_f1"]:.3f} ')

    
    
    ##################################################################################################################################
    # Draw the result
    ##################################################################################################################################
    plt.title(f'{batch_size} batch-sized {datasets_name} with \n Best test F1Mi(accuracy)={test_result["test"]:.3f}, valid F1Mi={test_result["valid"]:.3f}')
    # plt.title(f'{batch_size} batch-sized {datasets_name} with Best test F1Mi(accuracy)={test_result["micro_f1"]:.3f}')
    plt.plot(losses, label="Contrastive loss")
    plt.legend()
    plt.savefig(f"./figures/result_{datasets_name}.jpg", dpi=300)
    plt.close()
    print("Plot!")
    torch.save(encoder_model, f"./model_params/encoder_mdl_{datasets_name}.pth")
    print("Save!")


    f.write(f' {batch_size} batch-sized {datasets_name} with Best F1Mi(accuracy)={test_result["test"]:.3f}, train F1Mi={test_result["train"]:.3f}, valid F1Mi={test_result["valid"]:.3f} \n\n')
    # f.write(f' {batch_size} batch-sized {datasets_name} with Best F1Mi(accuracy)={test_result["micro_f1"]:.3f}\n\n')
    f.close()

    return {"dataset": datasets_name, "b_test": before_test, \
        "b_valid": before_valid, "test": test_result['test'], "valid": test_result['valid']}
    # return {"dataset": datasets_name, "b_test": before_test, "test": test_result['micro_f1']}




if __name__ == '__main__':
    args = arg_parser()
    exp_summary_file = osp.join('.', args.brief)
    before_test_acc = []
    before_valid_acc = []
    test_acc = []
    valid_acc = []
    for _ in range(args.iter):
        result = run(args)
        before_valid_acc.append(result['b_valid'])
        before_test_acc.append(result['b_test'])
        valid_acc.append(result['valid'])
        test_acc.append(result['test'])
    with open(exp_summary_file, 'a+') as f:
        writter = csv.writer(f)
        writter.writerow([
            args.dataset, 
            # np.mean(before_valid_acc), 
            # np.std(before_valid_acc),
            np.mean(before_test_acc),
            # np.std(before_test_acc),
            np.mean(valid_acc),
            # np.std(valid_acc),
            # np.max(valid_acc),
            np.mean(test_acc),
            np.std(test_acc),
            np.min(test_acc),
            np.max(test_acc),
            np.mean(sorted(test_acc, reverse=True)[:5])
        ])
