import torch
from evaluator import get_split, LREvaluator, SVMEvaluator


def test(encoder_model, dataloader, args):
    encoder_model.eval()
    x = []
    y = []
    encoder_model.eval()
    with torch.no_grad():
        for data in dataloader:
            data = data.to(args.device)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            _, g = encoder_model(data.x, data.edge_index, data.batch)
            x.append(g)
            y.append(data.y)
            
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    # result = LREvaluator()(x, y, split)
    result = SVMEvaluator()(x, y, split)
    
    return result