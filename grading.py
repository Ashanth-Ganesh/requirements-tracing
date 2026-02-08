import numpy as np
from typing import Dict, Tuple

class TraceMatrixEvaluator:
    
    def __init__(self, ground_truth_file: str):
        """
        Initialize evaluator with ground truth file
        
        Args:
            ground_truth_file: Path to the ground truth trace matrix
        """
        self.ground_truth = self.load_trace_matrix(ground_truth_file)
        self.num_frs, self.num_nfrs = self.ground_truth.shape
        
    def load_trace_matrix(self, filepath: str) -> np.ndarray:
        """Load trace matrix from CSV file"""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Split by comma and skip the FR ID
                    parts = line.split(',')
                    row = [int(x) for x in parts[1:]]  # Skip FR1, FR2, etc.
                    data.append(row)
        return np.array(data)
    
    def calculate_metrics(self, predicted: np.ndarray) -> Dict[str, float]:
        """
        Calculate precision, recall, F1-score, and accuracy
        
        Args:
            predicted: Predicted trace matrix
            
        Returns:
            Dictionary containing all metrics
        """
        # Flatten matrices for comparison
        gt_flat = self.ground_truth.flatten()
        pred_flat = predicted.flatten()
        
        # Calculate True Positives, False Positives, False Negatives, True Negatives
        tp = np.sum((gt_flat == 1) & (pred_flat == 1))
        fp = np.sum((gt_flat == 0) & (pred_flat == 1))
        fn = np.sum((gt_flat == 1) & (pred_flat == 0))
        tn = np.sum((gt_flat == 0) & (pred_flat == 0))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy
        }
    
    def evaluate_variant(self, variant_file: str, variant_num: int) -> Dict[str, float]:
        """Evaluate a single variant"""
        print(f"\n{'='*80}")
        print(f"EVALUATING VARIANT {variant_num}")
        print(f"{'='*80}")
        
        predicted = self.load_trace_matrix(variant_file)
        
        # Verify dimensions match
        if predicted.shape != self.ground_truth.shape:
            raise ValueError(f"Shape mismatch! Ground truth: {self.ground_truth.shape}, "
                           f"Predicted: {predicted.shape}")
        
        metrics = self.calculate_metrics(predicted)
        
        # Print detailed results
        print(f"\nFile: {variant_file}")
        print(f"\n--- Confusion Matrix ---")
        print(f"True Positives (TP):   {metrics['true_positives']:3d}  (Correctly identified traces)")
        print(f"False Positives (FP):  {metrics['false_positives']:3d}  (Incorrectly identified traces)")
        print(f"False Negatives (FN):  {metrics['false_negatives']:3d}  (Missed traces)")
        print(f"True Negatives (TN):   {metrics['true_negatives']:3d}  (Correctly identified non-traces)")
        
        print(f"\n--- Performance Metrics ---")
        print(f"Precision: {metrics['precision']:.4f}  (Of all predicted traces, how many were correct?)")
        print(f"Recall:    {metrics['recall']:.4f}  (Of all actual traces, how many did we find?)")
        print(f"F1-Score:  {metrics['f1_score']:.4f}  (Harmonic mean of precision and recall)")
        print(f"Accuracy:  {metrics['accuracy']:.4f}  (Overall correctness)")
        
        # Show trace comparison details
        total_gt_traces = np.sum(self.ground_truth)
        total_pred_traces = np.sum(predicted)
        print(f"\n--- Trace Counts ---")
        print(f"Ground Truth Traces:  {total_gt_traces}")
        print(f"Predicted Traces:     {total_pred_traces}")
        print(f"Difference:           {total_pred_traces - total_gt_traces:+d}")
        
        return metrics
    
    def compare_variants(self, variant_files: list) -> None:
        """Compare all variants and determine the best one"""
        print("\n" + "#"*80)
        print("TRACE MATRIX EVALUATION")
        print("#"*80)
        print(f"\nGround Truth: {self.num_frs} FRs × {self.num_nfrs} NFRs")
        print(f"Total possible links: {self.num_frs * self.num_nfrs}")
        print(f"Actual traces in ground truth: {np.sum(self.ground_truth)}")
        
        all_metrics = {}
        
        # Evaluate each variant
        for i, variant_file in enumerate(variant_files, start=1):
            try:
                metrics = self.evaluate_variant(variant_file, i)
                all_metrics[i] = metrics
            except FileNotFoundError:
                print(f"\n⚠️  Warning: {variant_file} not found. Skipping...")
            except Exception as e:
                print(f"\n❌ Error evaluating {variant_file}: {e}")
        
        # Summary comparison
        if len(all_metrics) > 0:
            print("\n" + "="*80)
            print("SUMMARY COMPARISON")
            print("="*80)
            
            print(f"\n{'Variant':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12}")
            print("-"*80)
            
            for variant_num, metrics in all_metrics.items():
                print(f"Variant {variant_num}  "
                      f"{metrics['precision']:<12.4f} "
                      f"{metrics['recall']:<12.4f} "
                      f"{metrics['f1_score']:<12.4f} "
                      f"{metrics['accuracy']:<12.4f}")
            
            # Determine best variant
            print("\n" + "="*80)
            print("BEST VARIANT SELECTION")
            print("="*80)
            
            # Best by different metrics
            best_precision = max(all_metrics.items(), key=lambda x: x[1]['precision'])
            best_recall = max(all_metrics.items(), key=lambda x: x[1]['recall'])
            best_f1 = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])
            best_accuracy = max(all_metrics.items(), key=lambda x: x[1]['accuracy'])
            
            print(f"\n🏆 Best Precision:  Variant {best_precision[0]} ({best_precision[1]['precision']:.4f})")
            print(f"🏆 Best Recall:     Variant {best_recall[0]} ({best_recall[1]['recall']:.4f})")
            print(f"🏆 Best F1-Score:   Variant {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
            print(f"🏆 Best Accuracy:   Variant {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
            
            print(f"\n{'='*80}")
            print(f"⭐ OVERALL BEST VARIANT: Variant {best_f1[0]} (F1-Score: {best_f1[1]['f1_score']:.4f})")
            print(f"{'='*80}")
            print("\nRationale: F1-Score is typically the best metric for trace link recovery")
            print("as it balances precision (avoiding false traces) and recall (finding all traces).")
            
            # Detailed recommendation
            self.print_recommendation(all_metrics, best_f1[0])
    
    def print_recommendation(self, all_metrics: Dict, best_variant: int):
        """Print detailed recommendation"""
        print("\n" + "="*80)
        print("DETAILED ANALYSIS")
        print("="*80)
        
        best = all_metrics[best_variant]
        
        print(f"\nVariant {best_variant} is recommended because:")
        
        if best['precision'] > 0.5 and best['recall'] > 0.5:
            print("✓ Good balance between precision and recall")
        elif best['precision'] > best['recall']:
            print(f"✓ High precision ({best['precision']:.4f}) - Few false positives")
            print(f"  ⚠️  Lower recall ({best['recall']:.4f}) - May miss some traces")
        else:
            print(f"✓ High recall ({best['recall']:.4f}) - Finds most traces")
            print(f"  ⚠️  Lower precision ({best['precision']:.4f}) - Some false positives")
        
        print(f"\nWith this variant:")
        print(f"  • You correctly identified {best['true_positives']} trace links")
        print(f"  • You missed {best['false_negatives']} trace links")
        print(f"  • You incorrectly added {best['false_positives']} trace links")


if __name__ == '__main__':
    # File paths
    ground_truth_file = 'ground_truth.txt'  # Your provided ground truth file
    variant_files = [
        'trace_matrix_1.txt',  # Variant 1: Lemmatization + stop-word removal
        'trace_matrix_2.txt',  # Variant 2: Stemming + stop-word removal
        'trace_matrix_3.txt'   # Variant 3: Stemming without stop-word removal
    ]
    
    # Create evaluator and compare
    evaluator = TraceMatrixEvaluator(ground_truth_file)
    evaluator.compare_variants(variant_files)