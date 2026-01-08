"""
SCRIPT CHẠY EVALUATION HOÀN CHỈNH

Quy trình:
1. Load models
2. Train-test split
3. Evaluate các methods
4. So sánh strategies
5. Visualize results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from main import HybridSystemManager
from evaluation import RecommenderEvaluator
import warnings

warnings.filterwarnings('ignore')


def run_full_evaluation(n_test_users: int = 100, k_values: list = [5, 10, 20]):
    """
    Chạy evaluation đầy đủ

    Args:
        n_test_users: Số lượng users để test (để nhanh hơn, set 100-500)
        k_values: Danh sách K values để test
    """

    print("=" * 70)
    print("FULL EVALUATION PIPELINE")
    print("=" * 70)

    # ===== STEP 1: Load System =====
    print("\n[STEP 1] Loading Hybrid System...")
    manager = HybridSystemManager()

    try:
        manager.load_all_models()
    except FileNotFoundError:
        print("Models not found. Building from scratch...")
        manager.build_all_models()

    # ===== STEP 2: Load Ratings =====
    print("\n[STEP 2] Loading ratings...")
    ratings = pd.read_csv('ml-latest/ratings.csv')
    print(f"Total ratings: {len(ratings):,}")

    # ===== STEP 3: Create Evaluator =====
    print("\n[STEP 3] Creating evaluator...")
    evaluator = RecommenderEvaluator(manager, ratings)

    # ===== STEP 4: Train-Test Split =====
    print("\n[STEP 4] Train-Test Split...")
    train_ratings, test_ratings, test_users = evaluator.train_test_split(
        test_size=0.2,
        min_ratings_per_user=10,
        random_state=42
    )

    # Giới hạn số test users nếu cần (để evaluation nhanh hơn)
    if len(test_users) > n_test_users:
        print(f"\nLimiting to {n_test_users} test users for faster evaluation...")
        test_users = test_users[:n_test_users]

    # ===== STEP 5: Comprehensive Evaluation =====
    print("\n[STEP 5] Running comprehensive evaluation...")
    print(f"Testing on {len(test_users)} users with K values: {k_values}")

    results = evaluator.comprehensive_evaluation(
        test_users=test_users,
        test_ratings=test_ratings,
        methods=['content', 'collab', 'hybrid'],
        k_values=k_values
    )

    # Save results
    results.to_csv('models/evaluation_results.csv', index=False)
    print("\n✅ Results saved to models/evaluation_results.csv")

    # ===== STEP 6: Compare Strategies =====
    print("\n[STEP 6] Comparing hybrid strategies...")

    strategy_results = evaluator.compare_strategies(
        test_users=test_users,
        test_ratings=test_ratings,
        strategies=['weighted', 'adaptive', 'switching'],
        k=10
    )

    strategy_results.to_csv('models/strategy_comparison.csv', index=False)
    print("\n✅ Strategy results saved to models/strategy_comparison.csv")

    # ===== STEP 7: Visualization =====
    print("\n[STEP 7] Creating visualizations...")
    visualize_results(results, strategy_results)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)

    return results, strategy_results


def visualize_results(results: pd.DataFrame, strategy_results: pd.DataFrame):
    """
    Tạo các biểu đồ trực quan
    """

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)

    # ===== FIGURE 1: Method Comparison (K=10) =====
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Method Comparison (K=10)', fontsize=16, fontweight='bold')

    # Filter K=10
    k10_data = results[results['k'] == 10].copy()

    # Plot 1: Precision & Recall
    ax = axes[0, 0]
    x = np.arange(len(k10_data))
    width = 0.35

    ax.bar(x - width / 2, k10_data['precision'], width, label='Precision', alpha=0.8)
    ax.bar(x + width / 2, k10_data['recall'], width, label='Recall', alpha=0.8)
    ax.set_xlabel('Method')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall @ K=10')
    ax.set_xticks(x)
    ax.set_xticklabels(k10_data['method'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Diversity
    ax = axes[0, 1]
    ax.bar(k10_data['method'], k10_data['diversity'], alpha=0.8, color='green')
    ax.set_xlabel('Method')
    ax.set_ylabel('Diversity Score')
    ax.set_title('Intra-List Diversity @ K=10')
    ax.grid(True, alpha=0.3)

    # Plot 3: Coverage
    ax = axes[1, 0]
    ax.bar(k10_data['method'], k10_data['catalog_coverage'], alpha=0.8, color='orange')
    ax.set_xlabel('Method')
    ax.set_ylabel('Coverage')
    ax.set_title('Catalog Coverage @ K=10')
    ax.grid(True, alpha=0.3)

    # Plot 4: Novelty
    ax = axes[1, 1]
    ax.bar(k10_data['method'], k10_data['novelty'], alpha=0.8, color='purple')
    ax.set_xlabel('Method')
    ax.set_ylabel('Novelty')
    ax.set_title('Novelty @ K=10')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('models/method_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: models/method_comparison.png")

    # ===== FIGURE 2: K Value Impact (Hybrid only) =====
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Impact of K Value (Hybrid Method)', fontsize=16, fontweight='bold')

    hybrid_data = results[results['method'] == 'hybrid'].sort_values('k')

    # Plot 1: Precision & Recall vs K
    ax = axes[0, 0]
    ax.plot(hybrid_data['k'], hybrid_data['precision'], marker='o', label='Precision', linewidth=2)
    ax.plot(hybrid_data['k'], hybrid_data['recall'], marker='s', label='Recall', linewidth=2)
    ax.set_xlabel('K Value')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: F1 Score vs K
    ax = axes[0, 1]
    ax.plot(hybrid_data['k'], hybrid_data['f1'], marker='o', color='red', linewidth=2)
    ax.set_xlabel('K Value')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs K')
    ax.grid(True, alpha=0.3)

    # Plot 3: Diversity vs K
    ax = axes[1, 0]
    ax.plot(hybrid_data['k'], hybrid_data['diversity'], marker='o', color='green', linewidth=2)
    ax.set_xlabel('K Value')
    ax.set_ylabel('Diversity')
    ax.set_title('Diversity vs K')
    ax.grid(True, alpha=0.3)

    # Plot 4: Coverage vs K
    ax = axes[1, 1]
    ax.plot(hybrid_data['k'], hybrid_data['catalog_coverage'], marker='o', color='orange', linewidth=2)
    ax.set_xlabel('K Value')
    ax.set_ylabel('Coverage')
    ax.set_title('Catalog Coverage vs K')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('models/k_value_impact.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: models/k_value_impact.png")

    # ===== FIGURE 3: Strategy Comparison =====
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Hybrid Strategy Comparison', fontsize=16, fontweight='bold')

    metrics = ['precision', 'recall', 'f1', 'diversity', 'coverage', 'novelty']
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        ax.bar(strategy_results['strategy'], strategy_results[metric],
               alpha=0.8, color=color)
        ax.set_xlabel('Strategy')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()}')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('models/strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: models/strategy_comparison.png")

    # ===== FIGURE 4: Radar Chart - Overall Comparison =====
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Normalize metrics to [0, 1]
    k10_normalized = k10_data.copy()
    metrics_to_plot = ['precision', 'recall', 'diversity', 'coverage', 'novelty']

    for metric in metrics_to_plot:
        max_val = k10_normalized[metric].max()
        if max_val > 0:
            k10_normalized[metric] = k10_normalized[metric] / max_val

    # Angles
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot

    # Plot each method
    colors = ['blue', 'green', 'red']
    for method, color in zip(['content', 'collab', 'hybrid'], colors):
        values = k10_normalized[k10_normalized['method'] == method][metrics_to_plot].values[0].tolist()
        values += values[:1]  # Close the plot

        ax.plot(angles, values, 'o-', linewidth=2, label=method.capitalize(), color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in metrics_to_plot])
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance Comparison (Normalized)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('models/radar_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: models/radar_comparison.png")

    plt.close('all')


def quick_test():
    """
    Quick test với ít users hơn (để test nhanh)
    """
    print("=" * 70)
    print("QUICK TEST MODE")
    print("=" * 70)

    results, strategy_results = run_full_evaluation(
        n_test_users=50,  # Chỉ test 50 users
        k_values=[10]  # Chỉ test K=10
    )

    return results, strategy_results


def detailed_analysis(results_df: pd.DataFrame):
    """
    Phân tích chi tiết kết quả
    """
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS")
    print("=" * 70)

    # Best method cho mỗi metric
    k10_data = results_df[results_df['k'] == 10]

    metrics = ['precision', 'recall', 'f1', 'diversity', 'coverage', 'novelty']

    print("\n📊 Best Method for Each Metric (K=10):")
    print("-" * 70)

    for metric in metrics:
        best_row = k10_data.loc[k10_data[metric].idxmax()]
        print(f"{metric.capitalize():15s}: {best_row['method']:10s} = {best_row[metric]:.4f}")

    # Trade-offs analysis
    print("\n⚖️  Trade-offs Analysis:")
    print("-" * 70)

    for method in ['content', 'collab', 'hybrid']:
        row = k10_data[k10_data['method'] == method].iloc[0]

        print(f"\n{method.upper()}:")
        print(f"  Accuracy (F1):      {row['f1']:.4f}")
        print(f"  Diversity:          {row['diversity']:.4f}")
        print(f"  Coverage:           {row['coverage']:.4f}")
        print(f"  Novelty:            {row['novelty']:.4f}")

        # Overall score (weighted average)
        overall = (
                0.4 * row['f1'] +
                0.3 * row['diversity'] +
                0.2 * row['coverage'] +
                0.1 * row['novelty']
        )
        print(f"  Overall Score:      {overall:.4f}")

    # Recommendations
    print("\n💡 Recommendations:")
    print("-" * 70)

    hybrid_row = k10_data[k10_data['method'] == 'hybrid'].iloc[0]

    if hybrid_row['f1'] > 0.15:
        print("✅ System has good accuracy (F1 > 0.15)")
    else:
        print("⚠️  Consider improving accuracy")

    if hybrid_row['diversity'] > 0.5:
        print("✅ Good diversity in recommendations")
    else:
        print("⚠️  Consider increasing diversity")

    if hybrid_row['coverage'] > 0.1:
        print("✅ Good catalog coverage")
    else:
        print("⚠️  Many movies are not being recommended (long-tail problem)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # Quick test mode
        results, strategy_results = quick_test()
    else:
        # Full evaluation
        results, strategy_results = run_full_evaluation(
            n_test_users=100,  # Thay đổi số này để điều chỉnh
            k_values=[5, 10, 20]
        )

    # Detailed analysis
    detailed_analysis(results)

    print("\n" + "=" * 70)
    print("ALL DONE! 🎉")
    print("=" * 70)
    print("\nGenerated files:")
    print("  📄 models/evaluation_results.csv")
    print("  📄 models/strategy_comparison.csv")
    print("  📊 models/method_comparison.png")
    print("  📊 models/k_value_impact.png")
    print("  📊 models/strategy_comparison.png")
    print("  📊 models/radar_comparison.png")