import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_score, recall_score, f1_score, precision_recall_curve, 
    roc_curve, auc
)
from imblearn.over_sampling import SMOTE
import matplotlib as mpl
import warnings

def run_pipeline():
    warnings.filterwarnings('ignore')

    base_path = r"C:\vs files\Aug 2026\2026-01-KIT-GS-ST-199"
    results_dir = os.path.join(base_path, "results")
    os.makedirs(results_dir, exist_ok=True)

    # --- 1. Predefined styling ---
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'

    print("Pipeline started...")

    # --- 2. Data Collection ---
    file_path = os.path.join(base_path, r"archive (66)\ITQMS_Dataset.csv")
    print(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)

    if 'Record_ID' in df.columns:
        df = df.drop(columns=['Record_ID'])

    target_col = 'Teaching_Quality_Label'
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # --- 3. Data Preprocessing ---
    print("Preprocessing data...")
    num_imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    if len(categorical_cols) > 0:
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    if df[target_col].isnull().any():
        df[target_col] = df[target_col].fillna(df[target_col].mode()[0])

    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    target_le = LabelEncoder()
    df[target_col] = target_le.fit_transform(df[target_col])

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --- 4. SMOTE & Train-Test Split ---
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

    # --- 5. Model Development ---
    print("Training Random Forest Classifier...")
    rf_classifier = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=2, 
        min_samples_leaf=1, random_state=42, n_jobs=-1, class_weight='balanced'
    )
    rf_classifier.fit(X_train, y_train)

    # --- 6. Evaluation Calculation ---
    print("Evaluating model...")
    y_pred = rf_classifier.predict(X_test)
    y_score = rf_classifier.predict_proba(X_test)
    cm = confusion_matrix(y_test, y_pred)
    target_names = target_le.inverse_transform(np.unique(y_resampled))
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

    # --- 7. Plotting Confusion Matrix ---
    print("Generating plots...")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=True, ax=ax, 
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 16, "weight": "bold", "family": "Times New Roman"})
    ax.set_title('Confusion Matrix', fontsize=22, fontweight='bold', fontfamily='Times New Roman', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=18, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    ax.set_ylabel('Actual Label', fontsize=18, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight("bold")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight("bold")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 8. Plotting Feature Importance ---
    importances = rf_classifier.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    features = X.columns[sorted_idx]
    sorted_importances = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=sorted_importances, y=features, hue=features, palette='Reds_r', ax=ax, legend=False)
    ax.set_title('Feature Importance', fontsize=22, fontweight='bold', fontfamily='Times New Roman', pad=20)
    ax.set_xlabel('Importance Score', fontsize=18, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    ax.set_ylabel('Features', fontsize=18, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight("bold")
    ax.grid(False)
    for i, v in enumerate(sorted_importances):
        ax.text(v + 0.005, i + 0.15, f"{v:.3f}", color='black', fontweight='bold', fontsize=16, fontfamily='Times New Roman')
    ax.set_xlim(0, max(sorted_importances) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 9. Plotting Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
    recall = np.insert(recall, 0, 1.0)
    precision = np.insert(precision, 0, 0.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(recall, precision, color='orange', lw=2)
    ax.set_title('Precision-Recall Curve', fontsize=22, fontweight='bold', fontfamily='Times New Roman', pad=20)
    ax.set_xlabel('Recall', fontsize=18, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    ax.set_ylabel('Precision', fontsize=18, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.tick_params(axis='both', which='major', labelsize=16)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight("bold")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 10. Plotting Overall Performance Metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [acc, prec, rec, f1]
    colors = ['#663399', '#2E8B57', '#FFA500', '#800080']

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.bar(metrics, values, color=colors, width=0.6)
    ax.set_title('Performance Metrics', fontsize=26, fontweight='bold', fontfamily='Times New Roman', pad=20)
    ax.set_xlabel('Evaluation Metrics', fontsize=22, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    ax.set_ylabel('Score Value', fontsize=22, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    ax.set_ylim([0, 1.1])
    ax.tick_params(axis='both', which='major', labelsize=18)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight("bold")
    ax.grid(False)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f"{yval:.4f}", ha='center', va='bottom', 
                color='black', fontweight='bold', fontsize=16, fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 11. Plotting ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color='dodgerblue', lw=2, label=f'Micro-average ROC curve (area = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=22, fontweight='bold', fontfamily='Times New Roman', pad=20)
    ax.set_xlabel('False Positive Rate', fontsize=18, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    ax.set_ylabel('True Positive Rate', fontsize=18, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(axis='both', which='major', labelsize=16)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight("bold")
    ax.grid(False)
    ax.legend(loc="lower right", prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 16})
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 12. Plotting FNR vs FPR Curve ---
    fnr = 1 - tpr
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, fnr, color='crimson', lw=2, label='FNR vs FPR curve')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_title('False Negative Rate vs False Positive Rate', fontsize=22, fontweight='bold', fontfamily='Times New Roman', pad=20)
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=18, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    ax.set_ylabel('False Negative Rate (FNR)', fontsize=18, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.tick_params(axis='both', which='major', labelsize=16)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight("bold")
    ax.grid(False)
    ax.legend(loc="upper right", prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 16})
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fnr_vs_fpr.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 13. Plotting Overall FNR vs FPR Bar Plot ---
    FP = cm.sum(axis=0) - np.diag(cm) 
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    FPR_classes = np.divide(FP, (FP + TN), out=np.zeros_like(FP, dtype=float), where=(FP+TN)!=0)
    FNR_classes = np.divide(FN, (TP + FN), out=np.zeros_like(FN, dtype=float), where=(TP+FN)!=0)

    macro_fpr = np.mean(FPR_classes)
    macro_fnr = np.mean(FNR_classes)

    labels = ['Overall FNR', 'Overall FPR']
    vals = [macro_fnr, macro_fpr]
    colors_bar = ['#DC143C', '#1E90FF'] 

    fig, ax = plt.subplots(figsize=(8, 8))
    bars = ax.bar(labels, vals, color=colors_bar, width=0.5)
    ax.set_title('Overall FNR vs FPR', fontsize=26, fontweight='bold', fontfamily='Times New Roman', pad=20)
    ax.set_xlabel('Error Metric', fontsize=22, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    ax.set_ylabel('Rate Value', fontsize=22, fontweight='bold', fontfamily='Times New Roman', labelpad=15)
    ax.set_ylim([0, max(vals) * 1.3])
    ax.tick_params(axis='both', which='major', labelsize=18)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight("bold")
    ax.grid(False)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max(vals) * 0.02), f"{yval:.4f}", ha='center', va='bottom', 
                color='black', fontweight='bold', fontsize=16, fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fnr_fpr_barplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 14. Save CSV Tables ---
    print("Exporting CSVs...")
    # Hyperparameters
    hp_dict = {
        'Hyperparameter': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state', 'n_jobs', 'class_weight'],
        'Value': [300, 20, 2, 1, 42, -1, 'balanced']
    }
    pd.DataFrame(hp_dict).to_csv(os.path.join(results_dir, 'hyperparameters.csv'), index=False)

    # Metrics
    metrics_dict = {
        'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score', 'Macro False Positive Rate (FPR)', 'Macro False Negative Rate (FNR)'],
        'Value': [acc, prec, rec, f1, macro_fpr, macro_fnr]
    }
    pd.DataFrame(metrics_dict).to_csv(os.path.join(results_dir, 'overall_metrics.csv'), index=False)

    print("Pipeline complete. All outputs saved to 'results' directory.")

if __name__ == '__main__':
    run_pipeline()
