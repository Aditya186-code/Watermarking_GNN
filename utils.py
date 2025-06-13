import torch
import torch.nn.functional as F
import copy
import hashlib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def select_poisoning_nodes(model, data, num_classes, target_label=None):
    """Select one high-confidence node per class"""
    model.eval()
    device = data.x.device

    with torch.no_grad():
        output = model(data)
        probabilities = F.softmax(output, dim=1)
        predictions = output.max(1)[1]

    poisoning_nodes = []
    original_labels = []

    for class_label in range(num_classes):
        class_mask = (predictions == class_label)
        class_nodes = torch.where(class_mask)[0]

        if len(class_nodes) == 0:
            continue

        class_confidences = probabilities[class_nodes, class_label]
        top_idx = class_nodes[torch.argmax(class_confidences)].item()
        poisoning_nodes.append(top_idx)
        original_labels.append(class_label)

    if target_label is None:
        target_label = num_classes

    poisoning_indices = torch.tensor(poisoning_nodes, dtype=torch.long, device=device)
    target_labels = torch.full((len(poisoning_indices),), target_label, dtype=torch.long, device=device)

    return poisoning_indices, target_labels


def attach_trigger_graph(data, trigger_features, trigger_indices, target_labels=None):
    """Attach triggers to the graph"""
    new_data = copy.deepcopy(data)
    new_data.x = data.x.clone()
    new_data.x[trigger_indices] = trigger_features[trigger_indices]

    if len(trigger_indices) > 1:
        trigger_pairs = []
        for i in range(len(trigger_indices)):
            for j in range(i+1, len(trigger_indices)):
                trigger_pairs.append([trigger_indices[i], trigger_indices[j]])
                trigger_pairs.append([trigger_indices[j], trigger_indices[i]])

        if trigger_pairs:
            trigger_edges = torch.tensor(trigger_pairs, dtype=torch.long).t()
            new_data.edge_index = torch.cat([data.edge_index, trigger_edges], dim=1)

    self_loops = torch.stack([trigger_indices, trigger_indices])
    new_data.edge_index = torch.cat([new_data.edge_index, self_loops], dim=1)

    if target_labels is not None:
        new_data.y = data.y.clone()
        new_data.y[trigger_indices] = target_labels

    return new_data


def compute_metrics(y_true, y_pred):
    """Compute classification metrics"""
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    accuracy = accuracy_score(y_true_np, y_pred_np) * 100
    precision = precision_score(y_true_np, y_pred_np, average='weighted', zero_division=0) * 100
    recall = recall_score(y_true_np, y_pred_np, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_true_np, y_pred_np, average='weighted', zero_division=0) * 100

    return accuracy, precision, recall, f1


def verify_ownership(suspect_model, generator, original_data, trigger_indices, target_labels, threshold):
    """Ownership verification"""
    suspect_model.eval()
    generator.eval()

    with torch.no_grad():
        trigger_features = generator(original_data)
        verification_data = attach_trigger_graph(original_data, trigger_features, trigger_indices, target_labels)
        output = suspect_model(verification_data)
        pred = output[trigger_indices].max(1)[1]
        correct = (pred == target_labels).sum().item()
        accuracy = (correct / len(trigger_indices)) * 100

    return accuracy >= threshold


def evaluate_model(model, data, detailed=False):
    """Evaluate model on clean data"""
    model.eval()
    with torch.no_grad():
        out = model(data)
        original_classes = torch.unique(data.y).max().item() + 1
        out_original = out[:, :original_classes]
        pred = out_original.max(1)[1]

        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
        overall_acc = (pred == data.y).float().mean()

        if detailed:
            y_true = data.y[data.test_mask].cpu().numpy()
            y_pred = pred[data.test_mask].cpu().numpy()

            accuracy = accuracy_score(y_true, y_pred) * 100
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100

            return {
                'test_acc': test_acc.item(),
                'train_acc': train_acc.item(),
                'overall_acc': overall_acc.item(),
                'sklearn_accuracy': accuracy,
                'sklearn_precision': precision,
                'sklearn_recall': recall,
                'sklearn_f1': f1
            }

    return test_acc.item(), train_acc.item(), overall_acc.item()


def owner_id_to_tensor(owner_secret, num_features, device='cpu'):
    """Generate owner ID tensor from secret string"""
    hash_hex = hashlib.sha256(owner_secret.encode()).hexdigest()
    hash_int = int(hash_hex, 16)
    hash_bits = bin(hash_int)[2:].zfill(num_features)[:num_features]
    tensor = torch.tensor([float(bit) for bit in hash_bits],
                         dtype=torch.float32, device=device)
    return tensor