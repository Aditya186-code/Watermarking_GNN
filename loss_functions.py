import torch
import torch.nn.functional as F


def watermarking_loss(output, target, indices):
    return F.nll_loss(output[indices], target[indices])


def imperception_loss(x, trigger_indices, edge_index, lambda_imp=1.0):
    """Imperception loss with proper similarity computation"""
    row, col = edge_index
    loss = 0.0
    count = 0

    for idx in trigger_indices:
        neighbors = col[row == idx]
        if len(neighbors) > 0:
            trigger_feat = x[idx].unsqueeze(0)
            neighbor_feats = x[neighbors]
            cos_sim = F.cosine_similarity(trigger_feat, neighbor_feats, dim=1)
            loss -= cos_sim.mean()
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=x.device, requires_grad=True)
    return lambda_imp * (loss / count)


def regulation_loss(trigger_features, owner_id, lambda_reg=1.0):
    """Regulation loss with proper owner ID encoding"""
    sigmoid_feats = torch.sigmoid(trigger_features)
    if owner_id.dim() == 1:
        owner_id = owner_id.unsqueeze(0).expand_as(sigmoid_feats)
    bce_loss = F.binary_cross_entropy(sigmoid_feats, owner_id, reduction='mean')
    return lambda_reg * bce_loss


def trigger_loss(output, target_label, trigger_indices, lambda_trig=1.0):
    """Trigger loss for classification"""
    trigger_output = output[trigger_indices]
    target_tensor = torch.full((len(trigger_indices),), target_label,
                              dtype=torch.long, device=output.device)
    return lambda_trig * F.nll_loss(trigger_output, target_tensor)