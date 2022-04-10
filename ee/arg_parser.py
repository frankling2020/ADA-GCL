import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='AD-AG-CL')
    parser.add_argument('--iter', type=int, default=1, help='Iteration')
    parser.add_argument('--dataset', type=str, default='PROTEINS', help='Dataset')
    parser.add_argument('--batch_size', type=int, default=256, help="Batch Size")
    parser.add_argument('--hidden_dims', type=int, default=32, help="hidden dims")
    parser.add_argument('--num_layers', type=int, default=2, help="num of layers")
    parser.add_argument('--edge_dims', type=int, default=3, help="edge dims")
    parser.add_argument('--device', type=str, default='cuda:3', help='Device for training')
    parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="Weight decay for Adam")
    parser.add_argument('--epochs', type=int, default=10, help='Epochs')
    parser.add_argument('--alpha', type=float, default=0.5, help='adversarial attack weight')
    parser.add_argument('--tau', type=float, default=0.2, help='Temperature')
    parser.add_argument('--reg', type=float, default=0.0, help='L2 regularization')
    parser.add_argument('--log', type=str, default='log_test.txt', help='Record the training results')
    parser.add_argument('--brief', type=str, default='summary.csv', help='Summarize')
    return parser.parse_args()