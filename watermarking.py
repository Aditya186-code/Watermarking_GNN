import torch
import torch.nn.functional as F
import copy
import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from models import get_model, TriggerGenerator
from loss_functions import imperception_loss, regulation_loss, trigger_loss
from utils import (select_poisoning_nodes, attach_trigger_graph, evaluate_model, 
                   verify_ownership, owner_id_to_tensor)


def run_watermarking(dataset_name, model_type='GCN', nodes_per_class=1, hidden_dim=64,
                    lr=0.01, outer_loops=50, inner_loops=10, verification_threshold=80.0):
    
    print(f"🚀 Starting watermarking process for {dataset_name}")
    print(f"Model: {model_type}, Hidden dim: {hidden_dim}, LR: {lr}")
    print("="*60)
    
    # Load dataset
    dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name,
                       transform=T.NormalizeFeatures())
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    print(f"📊 Dataset loaded: {dataset_name}")
    print(f"   - Nodes: {data.num_nodes}")
    print(f"   - Features: {dataset.num_features}")
    print(f"   - Classes: {dataset.num_classes}")
    print(f"   - Edges: {data.num_edges}")
    print(f"   - Device: {device}")
    print()

    # Initialize model
    model = get_model(model_type, dataset.num_features, hidden_dim, dataset.num_classes + 1, device)
    print(f"🧠 Model initialized: {model_type}")

    # Pre-train the clean model
    print("🔄 Pre-training clean model...")
    optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    model.train()
    for epoch in range(50):
        optimizer_pretrain.zero_grad()
        out = model(data)
        out_original = out[:, :dataset.num_classes]
        loss = F.nll_loss(out_original[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer_pretrain.step()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                model.eval()
                pred = out_original.argmax(dim=1)
                train_acc = pred[data.train_mask].eq(data.y[data.train_mask]).float().mean().item()
                test_pred = model(data)[:, :dataset.num_classes].argmax(dim=1)
                test_acc = test_pred[data.test_mask].eq(data.y[data.test_mask]).float().mean().item()
                model.train()
            print(f"   Epoch {epoch+1:2d}: Loss={loss.item():.4f}, Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")
    
    print("✅ Pre-training completed!")
    print()

    # Initialize generator
    generator = TriggerGenerator(dataset.num_features, hidden_dim).to(device)
    print("🎯 Trigger generator initialized")

    # Select poisoning nodes with detailed printing
    print("🎯 Selecting poisoning nodes...")
    trigger_indices, target_labels = select_poisoning_nodes(
        model, data, dataset.num_classes, target_label=dataset.num_classes)
    
    # Print selected nodes for each class
    print(f"Selected {len(trigger_indices)} trigger nodes:")
    
    # Group trigger indices by their original labels
    with torch.no_grad():
        model.eval()
        out = model(data)[:, :dataset.num_classes]
        pred_labels = out.argmax(dim=1)
        model.train()
    
    # Organize by original class
    class_to_nodes = {}
    for i, node_idx in enumerate(trigger_indices):
        original_label = data.y[node_idx].item()
        predicted_label = pred_labels[node_idx].item()
        if original_label not in class_to_nodes:
            class_to_nodes[original_label] = []
        class_to_nodes[original_label].append({
            'node_idx': node_idx.item(),
            'predicted': predicted_label,
            'target': target_labels[i].item()
        })
    
    for class_label in sorted(class_to_nodes.keys()):
        nodes_info = class_to_nodes[class_label]
        node_indices = [info['node_idx'] for info in nodes_info]
        print(f"   Class {class_label}: nodes {node_indices}")
        for info in nodes_info:
            print(f"      Node {info['node_idx']}: pred={info['predicted']} → target={info['target']}")
    print()

    # Optimizers for bi-level optimization
    optimizer_model = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr*0.5)

    # Schedulers
    scheduler_model = torch.optim.lr_scheduler.StepLR(optimizer_model, step_size=20, gamma=0.8)
    scheduler_gen = torch.optim.lr_scheduler.StepLR(optimizer_generator, step_size=20, gamma=0.8)

    # Generate owner ID
    owner_id = owner_id_to_tensor("YourUniqueOwnerSecret", dataset.num_features, device)
    owner_id = owner_id.unsqueeze(0).expand(len(trigger_indices), -1)
    print(f"🔐 Owner ID generated (shape: {owner_id.shape})")
    print()

    # Training loop
    print("🚀 Starting bi-level optimization...")
    print("Epoch | Train Acc | Test Acc  | Embed Loss | Gen Loss  | LR_model | LR_gen")
    print("-" * 75)
    
    model.train()
    generator.train()

    best_test_acc = 0.0
    epochs = []
    test_accuracies = []
    train_accuracies = []
    embedding_losses = []
    generator_losses = []

    for outer in range(outer_loops):
        # Inner-level optimization
        inner_losses = []
        for inner in range(inner_loops):
            optimizer_model.zero_grad()

            with torch.no_grad():
                triggered_features = generator(data)

            data_triggered = attach_trigger_graph(data, triggered_features, trigger_indices, target_labels)
            output = model(data_triggered)

            clean_output = output[:, :dataset.num_classes]
            clean_loss = F.nll_loss(clean_output[data.train_mask], data.y[data.train_mask])
            watermark_loss = F.nll_loss(output[trigger_indices], target_labels)

            L_embed = clean_loss + 0.8 * watermark_loss
            L_embed.backward()
            optimizer_model.step()
            inner_losses.append(L_embed.item())

        # Store model snapshot
        model_snapshot = copy.deepcopy(model)
        model_snapshot.eval()

        # Outer-level optimization
        optimizer_generator.zero_grad()

        triggered_features = generator(data)
        data_triggered = attach_trigger_graph(data, triggered_features, trigger_indices)

        with torch.no_grad():
            output_eval = model_snapshot(data_triggered)

        loss_imp = imperception_loss(data_triggered.x, trigger_indices, data_triggered.edge_index)
        loss_reg = regulation_loss(triggered_features[trigger_indices], owner_id)
        loss_trig = trigger_loss(output_eval, target_labels[0].item(), trigger_indices)

        epsilon_1, epsilon_2, epsilon_3 = 0.3, 0.3, 0.4
        L_g = epsilon_1 * loss_imp + epsilon_2 * loss_reg + epsilon_3 * loss_trig

        L_g.backward()
        optimizer_generator.step()

        scheduler_model.step()
        scheduler_gen.step()

        # Track metrics
        test_acc, train_acc, overall_acc = evaluate_model(model, data)
        epochs.append(outer + 1)
        test_accuracies.append(test_acc)
        train_accuracies.append(train_acc)
        embedding_losses.append(np.mean(inner_losses))
        generator_losses.append(L_g.item())

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # Print progress every 10 epochs
        if (outer + 1) % 10 == 0 or outer == 0:
            current_lr_model = optimizer_model.param_groups[0]['lr']
            current_lr_gen = optimizer_generator.param_groups[0]['lr']
            print(f"{outer+1:5d} | {train_acc:8.3f} | {test_acc:8.3f} | {np.mean(inner_losses):9.4f} | "
                  f"{L_g.item():8.4f} | {current_lr_model:.2e} | {current_lr_gen:.2e}")

    print("-" * 75)
    print(f"✅ Training completed! Best test accuracy: {best_test_acc:.3f}")
    print()

    # Final evaluation
    print("📊 Final evaluation...")
    detailed_results = evaluate_model(model, data, detailed=True)
    final_precision = detailed_results['sklearn_precision']
    final_recall = detailed_results['sklearn_recall']
    final_f1 = detailed_results['sklearn_f1']

    final_test_acc = detailed_results['test_acc']
    
    print(f"Final results:")
    print(f"   - Test accuracy: {final_test_acc:.3f}")
    print(f"   - Train accuracy: {detailed_results.get('train_acc', 'N/A')}")
    if 'overall_acc' in detailed_results:
        print(f"   - Overall accuracy: {detailed_results['overall_acc']:.3f}")
    print()

    # Calculate final watermark accuracy
    print("🎯 Calculating watermark accuracy...")
    model.eval()
    generator.eval()
    
    with torch.no_grad():
        # Generate triggered features
        triggered_features = generator(data)
        data_triggered = attach_trigger_graph(data, triggered_features, trigger_indices, target_labels)
        
        # Get model predictions on triggered data
        output = model(data_triggered)
        watermark_predictions = output[trigger_indices].argmax(dim=1)
        
        # Calculate watermark accuracy
        watermark_correct = watermark_predictions.eq(target_labels).sum().item()
        watermark_accuracy = (watermark_correct / len(trigger_indices)) * 100.0
    
    print(f"Watermark accuracy: {watermark_accuracy:.1f}% ({watermark_correct}/{len(trigger_indices)})")
    print()

    # Ownership verification
    print("🔍 Verifying ownership...")
    is_verified = verify_ownership(model, generator, data, trigger_indices,
                                 target_labels, verification_threshold)
    
    verification_status = "✅ VERIFIED" if is_verified else "❌ NOT VERIFIED"
    print(f"Ownership verification: {verification_status}")
    print(f"   - Threshold: {verification_threshold}%")
    print(f"   - Actual watermark accuracy: {watermark_accuracy:.1f}%")
    print()
    
    # Summary
    print("📋 SUMMARY")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_type}")
    print(f"Trigger nodes: {len(trigger_indices)}")
    print(f"Final test accuracy: {final_test_acc:.3f}")
    print(f"Best test accuracy: {best_test_acc:.3f}")
    print(f"Watermark accuracy: {watermark_accuracy:.1f}%")
    print(f"Ownership verified: {is_verified}")
    print("="*60)

    return {
    "model": model,
    "generator": generator,
    "final_test_acc": final_test_acc,
    "is_verified": is_verified,
    "epochs": epochs,
    "test_accuracies": test_accuracies,
    "train_accuracies": train_accuracies,
    "embedding_losses": embedding_losses,
    "generator_losses": generator_losses,
    "watermark_accuracy": watermark_accuracy,
    "final_precision": final_precision,
    "final_recall": final_recall,
    "final_f1": final_f1
}
