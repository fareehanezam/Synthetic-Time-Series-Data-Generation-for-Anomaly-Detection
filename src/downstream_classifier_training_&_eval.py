
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Import pandas for DataFrame creation
import json # Import json for saving reports
import os # Import os for path joining
import joblib # Import joblib for saving models
from IPython.display import display # Import display

from src.logger import get_logger # Import the logger initialization function
logger = get_logger("DownstreamClassifierEvaluation") # Initialize logger at the beginning

logger.info("Starting downstream model training and evaluation.")

# --- Utility function for training and evaluation ---
def train_and_evaluate(df, model_name):
    logger.info(f"--- Evaluating {model_name} ---")

    # Create a clean version of the model name for filenames
    model_name_clean = model_name.replace(" ", "_").replace("(", "").replace(")", "")

    # Label encode the target variable
    le = LabelEncoder()
    df['machine_status'] = le.fit_transform(df['machine_status'])

    X = df.drop('machine_status', axis=1)
    y = df['machine_status']

    # Ensure sufficient samples for stratification in train_test_split
    if len(y.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    else:
        # Handle cases with only one class 
        logger.warning(f"Only one class present in the data for {model_name}. Cannot perform stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    model = XGBClassifier(objective='multi:softmax', num_class=len(le.classes_), use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Get predicted probabilities for ROC curve
    y_pred_proba = model.predict_proba(X_test)


    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    logger.info(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    # Convert class names to strings for labels
    class_labels = [str(cls) for cls in le.classes_]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join('results', f'cm_{model_name_clean}.png'), dpi=300)
    plt.show()

    # Plot Feature Importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_importance)), feature_importance['importance'].values)
        plt.yticks(range(len(feature_importance)), feature_importance['feature'].values)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top 15 Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join('results', f'feature_importance_{model_name_clean}.png'), dpi=300)
        plt.show()

    return report, [str(cls) for cls in le.classes_], model, X_test, y_test, y_pred_proba

def plot_roc_curves(baseline_data, augmented_data, class_names, target_class='RECOVERING'):
    """Plot ROC curves comparing baseline and augmented models"""
    logger.info("Plotting ROC curves for model comparison.")

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Binarize the labels for multi-class ROC
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    y_test_baseline_bin = label_binarize(baseline_data['y_test'], classes=range(len(class_names)))
    y_test_augmented_bin = label_binarize(augmented_data['y_test'], classes=range(len(class_names)))

    # Find the index of the target class
    if target_class in class_names:
        target_idx = class_names.index(target_class)

        plt.figure(figsize=(10, 8))

        # Baseline ROC
        # Ensure y_pred_proba_baseline has probabilities for all classes, even if some are zero
        if baseline_data['y_pred_proba'].shape[1] < len(class_names):
             logger.error("Baseline model did not output probabilities for all classes. Cannot plot ROC.")
             print("Baseline model did not output probabilities for all classes. Cannot plot ROC.")
             return


        fpr_baseline, tpr_baseline, _ = roc_curve(y_test_baseline_bin[:, target_idx],
                                                    baseline_data['y_pred_proba'][:, target_idx])
        roc_auc_baseline = auc(fpr_baseline, tpr_baseline)

        # Augmented ROC
        if augmented_data['y_pred_proba'].shape[1] < len(class_names):
             logger.error("Augmented model did not output probabilities for all classes. Cannot plot ROC.")
             print("Augmented model did not output probabilities for all classes. Cannot plot ROC.")
             return

        fpr_augmented, tpr_augmented, _ = roc_curve(y_test_augmented_bin[:, target_idx],
                                                     augmented_data['y_pred_proba'][:, target_idx])
        roc_auc_augmented = auc(fpr_augmented, tpr_augmented)

        plt.plot(fpr_baseline, tpr_baseline, label=f'Baseline (AUC = {roc_auc_baseline:.3f})',
                linewidth=2, color='blue')
        plt.plot(fpr_augmented, tpr_augmented, label=f'Augmented (AUC = {roc_auc_augmented:.3f})',
                linewidth=2, color='green')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve Comparison - "{target_class}" Class')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join('results', f'roc_curve_{target_class.lower()}.png'), dpi=300)
        plt.show()

        logger.info(f"ROC curves plotted for '{target_class}' class.")
    else:
        logger.warning(f"Target class '{target_class}' not found in class names.")

def plot_class_distribution(original_df, synthetic_df, augmented_df):
    """Plot class distribution comparison"""
    logger.info("Plotting class distribution.")

    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original data distribution
    original_counts = original_df['machine_status'].value_counts()
    axes[0].bar(range(len(original_counts)), original_counts.values, color='steelblue')
    axes[0].set_xticks(range(len(original_counts)))
    axes[0].set_xticklabels(original_counts.index, rotation=45, ha='right')
    axes[0].set_title('Original Data Distribution')
    axes[0].set_xlabel('Machine Status')
    axes[0].set_ylabel('Count')

    # Synthetic data distribution
    if not synthetic_df.empty:
        synthetic_counts = synthetic_df['machine_status'].value_counts()
        axes[1].bar(range(len(synthetic_counts)), synthetic_counts.values, color='coral')
        axes[1].set_xticks(range(len(synthetic_counts)))
        axes[1].set_xticklabels(synthetic_counts.index, rotation=45, ha='right')
        axes[1].set_title('Synthetic Data Distribution')
        axes[1].set_xlabel('Machine Status')
        axes[1].set_ylabel('Count')
    else:
        axes[1].text(0.5, 0.5, 'No Synthetic Data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Synthetic Data Distribution')

    # Augmented data distribution
    augmented_counts = augmented_df['machine_status'].value_counts()
    axes[2].bar(range(len(augmented_counts)), augmented_counts.values, color='mediumseagreen')
    axes[2].set_xticks(range(len(augmented_counts)))
    axes[2].set_xticklabels(augmented_counts.index, rotation=45, ha='right')
    axes[2].set_title('Augmented Data Distribution')
    axes[2].set_xlabel('Machine Status')
    axes[2].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join('results', 'class_distribution_comparison.png'), dpi=300)
    plt.show()

    logger.info("Class distribution plot saved.")

def plot_metrics_heatmap(baseline_report, augmented_report, class_names):
    """Plot heatmap of metrics for all classes"""
    logger.info("Plotting metrics heatmap.")

    os.makedirs('results', exist_ok=True)

    metrics = ['precision', 'recall', 'f1-score']

    # Create data for heatmap
    baseline_data = [[baseline_report[cls][metric] for metric in metrics] for cls in class_names if cls in baseline_report]
    augmented_data = [[augmented_report[cls][metric] for metric in metrics] for cls in class_names if cls in augmented_report]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Baseline heatmap
    sns.heatmap(baseline_data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=metrics, yticklabels=class_names, ax=axes[0], vmin=0, vmax=1)
    axes[0].set_title('Baseline Model Metrics')
    axes[0].set_ylabel('Class')

    # Augmented heatmap
    sns.heatmap(augmented_data, annot=True, fmt='.3f', cmap='YlGn',
                xticklabels=metrics, yticklabels=class_names, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title('Augmented Model Metrics')
    axes[1].set_ylabel('Class')

    plt.tight_layout()
    plt.savefig(os.path.join('results', 'metrics_heatmap.png'), dpi=300)
    plt.show()

    logger.info("Metrics heatmap saved.")

def evaluate_classifiers(original_df, synthetic_df, target_class='RECOVERING'):
    logger.info("Starting classifier evaluation.")

    # Prepare datasets
    # Ensure 'timestamp' column is dropped if it exists
    if 'timestamp' in original_df.columns:
        original_df = original_df.drop(columns=['timestamp'])

    if 'timestamp' in synthetic_df.columns:
        synthetic_df = synthetic_df.drop(columns=['timestamp'])

    if not synthetic_df.empty:
        augmented_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    else:
        augmented_df = original_df.copy()

    # Plot class distribution
    plot_class_distribution(original_df.copy(), synthetic_df.copy(), augmented_df.copy())

    # 1. Baseline Model
    baseline_report, class_names, baseline_model, X_test_baseline, y_test_baseline, y_pred_proba_baseline = \
        train_and_evaluate(original_df.copy(), "Baseline (Original Data)")

    # Save Baseline Model
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(baseline_model, os.path.join(models_dir, 'xgboost_baseline_model.joblib'))
    logger.info("Baseline XGBoost model saved.")


    # 2. Augmented Model
    augmented_report, _, augmented_model, X_test_augmented, y_test_augmented, y_pred_proba_augmented = \
        train_and_evaluate(augmented_df.copy(), "Augmented (Synthetic Data)")

    # Save Augmented Model
    joblib.dump(augmented_model, os.path.join(models_dir, 'xgboost_augmented_model.joblib'))
    logger.info("Augmented XGBoost model saved.")


    # --- Compare Results ---
    print("\n--- Performance Comparison ---")
    logger.info("Comparing model performances.")

    # Ensure target_class is in the reports before accessing
    if target_class not in baseline_report or target_class not in augmented_report:
        logger.error(f"Target class '{target_class}' not found in classification reports. Cannot generate comparison.")
        print(f"Target class '{target_class}' not found in classification reports. Cannot generate comparison.")
        return

    comparison_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        f'Baseline_{target_class}': [
            baseline_report[target_class]['precision'],
            baseline_report[target_class]['recall'],
            baseline_report[target_class]['f1-score']
        ],
        f'Augmented_{target_class}': [
            augmented_report[target_class]['precision'],
            augmented_report[target_class]['recall'],
            augmented_report[target_class]['f1-score']
        ]
    }
    comparison_df = pd.DataFrame(comparison_data)

    print(f"Focusing on the '{target_class}' class:")
    display(comparison_df)
    logger.info(f"Comparison results:\n{comparison_df.to_string()}")

    # Calculate and display percentage improvement
    improvement_data = {
        'Metric': ['Precision Improvement (%)', 'Recall Improvement (%)', 'F1-Score Improvement (%)'],
        'Improvement': []
    }

    metrics_to_compare = ['precision', 'recall', 'f1-score']
    for metric in metrics_to_compare:
        baseline_score = baseline_report[target_class].get(metric, 0) # Use .get to handle missing keys gracefully
        augmented_score = augmented_report[target_class].get(metric, 0)

        if baseline_score != 0: # Avoid division by zero
            improvement = ((augmented_score - baseline_score) / baseline_score) * 100
            improvement_data['Improvement'].append(f"{improvement:.2f}%")
        elif augmented_score > 0: # If baseline is 0 but augmented is > 0, it's infinite improvement
             improvement_data['Improvement'].append("Infinite%")
        else: # If both are 0
             improvement_data['Improvement'].append("N/A")


    improvement_df = pd.DataFrame(improvement_data)
    print(f"\nPerformance Improvement for '{target_class}' Class:")
    display(improvement_df)
    logger.info(f"Improvement results:\n{improvement_df.to_string()}")


    # Plot comparison of key metrics for the target class
    metrics_to_plot = ['precision', 'recall', 'f1-score']
    comparison_plot_data = {
        'Metric': metrics_to_plot * 2,
        'Model': ['Baseline'] * len(metrics_to_plot) + ['Augmented'] * len(metrics_to_plot),
        'Score': [baseline_report[target_class][m] for m in metrics_to_plot] +
                 [augmented_report[target_class][m] for m in metrics_to_plot]
    }
    comparison_plot_df = pd.DataFrame(comparison_plot_data)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Score', hue='Model', data=comparison_plot_df, palette='viridis')
    plt.title(f'Performance Comparison for "{target_class}" Class')
    plt.ylabel('Score')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join('results', f'performance_comparison_{target_class.lower()}.png'), dpi=300)
    plt.show()
    logger.info(f"Performance comparison plot saved for '{target_class}' class.")

    # Plot ROC curves
    baseline_data = {
        'y_test': y_test_baseline,
        'y_pred_proba': y_pred_proba_baseline
    }
    augmented_data = {
        'y_test': y_test_augmented,
        'y_pred_proba': y_pred_proba_augmented
    }
    plot_roc_curves(baseline_data, augmented_data, class_names, target_class)

    # Plot metrics heatmap for all classes
    plot_metrics_heatmap(baseline_report, augmented_report, class_names)

    # Save reports to a file
    with open(os.path.join('results', 'evaluation_reports.json'), 'w') as f:
        json.dump({'baseline': baseline_report, 'augmented': augmented_report}, f, indent=4)
    logger.info("Evaluation reports saved to results/ directory.")
