import numpy as np
from watermarking import run_watermarking
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run watermarking experiments')
    parser.add_argument('command', nargs='?', help='Command: "all" to run all combinations')
    parser.add_argument('--model', type=str, help='Specific model to run (e.g., GCN)')
    parser.add_argument('--dataset', type=str, help='Specific dataset to run (e.g., Cora, Pubmed)')
    
    args = parser.parse_args()
    
    results = {}
    
    # Define all available models and datasets
    all_model_types = ['GCN', 'GAT', 'GraphSAGE']
    all_dataset_names = ['Cora', 'Pubmed']
    
    # Determine which models and datasets to run
    if args.command == 'all':
        # Run all combinations
        model_types = all_model_types
        dataset_names = all_dataset_names
        print("Running all model-dataset combinations...")
    elif args.model and args.dataset:
        # Run specific model-dataset combination
        if args.model not in all_model_types:
            print(f"Error: Model '{args.model}' not supported. Available models: {all_model_types}")
            return
        if args.dataset not in all_dataset_names:
            print(f"Error: Dataset '{args.dataset}' not supported. Available datasets: {all_dataset_names}")
            return
        model_types = [args.model]
        dataset_names = [args.dataset]
        print(f"Running {args.model} on {args.dataset}...")
    elif args.model:
        # Run specific model on all datasets
        if args.model not in all_model_types:
            print(f"Error: Model '{args.model}' not supported. Available models: {all_model_types}")
            return
        model_types = [args.model]
        dataset_names = all_dataset_names
        print(f"Running {args.model} on all datasets...")
    elif args.dataset:
        # Run all models on specific dataset
        if args.dataset not in all_dataset_names:
            print(f"Error: Dataset '{args.dataset}' not supported. Available datasets: {all_dataset_names}")
            return
        model_types = all_model_types
        dataset_names = [args.dataset]
        print(f"Running all models on {args.dataset}...")
    else:
        # No arguments provided - show error message
        print("❌ Error: No model or dataset specified!")
        print("\nUsage:")
        print("  python3 main.py all                           # Run all model-dataset combinations")
        print("  python3 main.py --model GCN --dataset Cora   # Run specific model on specific dataset")
        print("  python3 main.py --model GCN                  # Run specific model on all datasets")
        print("  python3 main.py --dataset Cora               # Run all models on specific dataset")
        print(f"\nAvailable models: {all_model_types}")
        print(f"Available datasets: {all_dataset_names}")
        return
    
    # Run experiments for selected dataset-model combinations
    for dataset_name in dataset_names:
        for model_type in model_types:
            try:
                print(f"Running {model_type} on {dataset_name}...")
                result = run_watermarking(
                    dataset_name=dataset_name,
                    model_type=model_type,
                    nodes_per_class=1,
                    hidden_dim=64,
                    lr=0.01,
                    outer_loops=50,
                    inner_loops=3,
                    verification_threshold=70.0
                )
                results[(dataset_name, model_type)] = result
                print(f"✅ Completed {model_type} on {dataset_name}")
            except Exception as e:
                print(f"❌ Error with {model_type} on {dataset_name}: {e}")
                continue

    # Generate performance table
    if results:
        output_lines = []

        # Add top-level table header
        output_lines.append("\n" + "="*80)
        output_lines.append("MODEL PERFORMANCE TABLE")
        output_lines.append("="*80)
        output_lines.append("Table: Model Performance (Watermarked Models)")
        output_lines.append("-" * 110)
        output_lines.append(f"{'Dataset':<10} {'Model':<12} {'Accuracy (%)':<12} {'Precision (%)':<14} {'Recall (%)':<12} {'F1-score (%)':<12} {'Verified':<10}")

        # Iterate through dataset results
        for dataset_name in dataset_names:
            dataset_results = {k: v for k, v in results.items() if k[0] == dataset_name}
            if dataset_results:
                first_model = True
                for model_key in sorted(dataset_results.keys(), key=lambda x: x[1]):
                    model_name = model_key[1]
                    data = dataset_results[model_key]
                    accuracy = data['final_test_acc'] * 100
                    is_verified = data['is_verified']
                    
                    # Estimate other metrics based on accuracy with small variations
                    precision = data['final_precision']
                    recall = data['final_recall']
                    f1 = data['final_f1']                    
                    # Ensure values are reasonable
                    precision = max(0, min(100, precision))
                    recall = max(0, min(100, recall))
                    f1 = max(0, min(100, f1))
                    
                    verified_status = "✅ Yes" if is_verified else "❌ No"
                    
                    if first_model:
                        line = f"{dataset_name:<10} {model_name:<12} {accuracy:<12.2f} {precision:<14.2f} {recall:<12.2f} {f1:<12.2f} {verified_status:<10}"
                        first_model = False
                    else:
                        line = f"{'':10} {model_name:<12} {accuracy:<12.2f} {precision:<14.2f} {recall:<12.2f} {f1:<12.2f} {verified_status:<10}"
                    
                    output_lines.append(line)

        # Add separate watermark accuracy tables for each dataset
        output_lines.append("\n" + "="*80)
        output_lines.append("WATERMARK ACCURACY TABLES")
        output_lines.append("="*80)
        
        for dataset_name in dataset_names:
            dataset_results = {k: v for k, v in results.items() if k[0] == dataset_name}
            if dataset_results:
                output_lines.append(f"\nTable: Watermark Accuracy - {dataset_name} Dataset")
                output_lines.append("-" * 60)
                output_lines.append(f"{'Model':<20} {'Watermark Accuracy (%)':<25} {'Verification Status':<15}")
                output_lines.append("-" * 60)
                
                for model_key in sorted(dataset_results.keys(), key=lambda x: x[1]):
                    model_name = model_key[1]
                    data = dataset_results[model_key]
                    watermark_accuracy = data['watermark_accuracy']  # Extract watermark accuracy
                    is_verified = data['is_verified']
                    verified_status = "✅ Verified" if is_verified else "❌ Failed"
                    
                    line = f"{model_name:<20} {watermark_accuracy:<25.2f} {verified_status:<15}"
                    output_lines.append(line)

        # Write to file
        with open("results_table.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))

    else:
        print("❌ No results to display - all experiments failed.")

if __name__ == "__main__":
    main()