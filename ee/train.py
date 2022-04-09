import torch
import seaborn as sns
from utils import normalize
from loss import loss_fn

def train(encoder_model, dataloader, optimizer, args):
    
    alpha = args.alpha
    reg = args.reg
    device = args.device
    
    encoder_model.train()
    epoch_loss = 0

    epochs = len(dataloader)
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        z, g = encoder_model(x, edge_index, batch)

        edge_attn, node_attn = encoder_model.attn_scores(z.detach(), g.detach(), batch, edge_index)

        gs = []
        rates = []
        for drop_rate in [0.3, 0.4]:
            x_aug, edge = encoder_model.learnable_transform(x, edge_index, edge_attn, node_attn, drop_rate)

            _, g_aug = encoder_model.encoder(x_aug, edge, batch)
            gs.append(g_aug)

            rates.append(edge.shape[1]/edge_index.shape[1])

        g1, g2 = gs[0], gs[1]

        t1 = encoder_model.mlp(g1)
        t2 = encoder_model.mlp(g2)
        
        g_normed = normalize(g)
        g_flag = torch.where(g_normed > 0.1, 0, 1)
        g1_pos = g_flag * g1.detach()

        t3 = g1_pos
        t4 = g2.detach()
        
        loss = loss_fn(t1, t2) - alpha * loss_fn(t3,t4, beta=0.0) + reg * encoder_model.reg_loss()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss/epochs, None