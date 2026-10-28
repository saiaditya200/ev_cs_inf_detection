import pandas as pd
import json

df = pd.read_csv('results/diagnostics/final_hyperparameter_search.csv')

print("=== 1. BASELINE REPRODUCTION ===")
# Random Forest, n_estimators=250, max_depth=3, min_samples_split=2, min_samples_leaf=2, max_features=0.5, class_weight=null, threshold=0.76
mask_base = (
    (df['model_family'] == 'Random_Forest') &
    (df['threshold'] == 0.76) &
    (df['hyperparameters'].str.contains('250')) &
    (df['hyperparameters'].str.contains('"max_depth": 3')) &
    (df['hyperparameters'].str.contains('"min_samples_split": 2')) &
    (df['hyperparameters'].str.contains('"min_samples_leaf": 2')) &
    (df['hyperparameters'].str.contains('"max_features": 0.5')) &
    (df['hyperparameters'].str.contains('"class_weight": null'))
)
base_row = df[mask_base]
if len(base_row) > 0:
    r = base_row.iloc[0]
    print(f"Mean CV Accuracy: {r['mean_cv_accuracy']*100:.2f}% (std: {r['std_cv_accuracy']*100:.2f}%)")
    print(f"Mean CV Balanced Accuracy: {r['mean_cv_balanced_accuracy']*100:.2f}% (std: {r['std_cv_balanced_accuracy']*100:.2f}%)")
    print(f"Class 0 Recall: {r['mean_cv_minority_recall']*100:.2f}% (std: {r['std_cv_minority_recall']*100:.2f}%)")
    print(f"Class 0 F1: {r['mean_cv_minority_f1']:.4f}")
    print(f"Macro F1: {r['mean_cv_macro_f1']:.4f}")
    print(f"PR-AUC: {r['mean_cv_pr_auc']:.4f}")
    print(f"ROC-AUC: {r['mean_cv_roc_auc']:.4f}")
    print(f"OOF Accuracy: {r['oof_accuracy']*100:.2f}%")
    print(f"OOF Balanced Accuracy: {r['oof_balanced_accuracy']*100:.2f}%")
    print(f"OOF Minority F1: {r['oof_minority_f1']:.4f}")
    print(f"OOF Macro F1: {r['oof_macro_f1']:.4f}")

print("\n=== 2. CLASS WEIGHT EXPERIMENTS ===")
# Check which class weights exist in search
print("Unique class weights in hyperparameters:")
for cw in ['null', '"balanced"']:
    sub = df[df['hyperparameters'].str.contains(f'"class_weight": {cw}')]
    print(f"Count for class_weight={cw}: {len(sub)}")
    best_bal = sub.sort_values(by=['mean_cv_balanced_accuracy', 'mean_cv_accuracy'], ascending=False).iloc[0]
    print(f"  Best for {cw}: Thresh={best_bal['threshold']}, Acc={best_bal['mean_cv_accuracy']*100:.2f}%, BalAcc={best_bal['mean_cv_balanced_accuracy']*100:.2f}%, C0_Rec={best_bal['mean_cv_minority_recall']*100:.2f}%, C0_F1={best_bal['mean_cv_minority_f1']:.4f}, PR_AUC={best_bal['mean_cv_pr_auc']:.4f}")
    print(f"  Params: {best_bal['hyperparameters']}")

print("\n=== 3. TOP 10 RANDOM FOREST CANDIDATES BY BALANCED ACCURACY ===")
rf_df = df[df['model_family'] == 'Random_Forest']
top_bal = rf_df.sort_values(by=['mean_cv_balanced_accuracy', 'mean_cv_accuracy'], ascending=False).head(10)
for idx, (_, row) in enumerate(top_bal.iterrows(), 1):
    print(f"{idx}. BalAcc: {row['mean_cv_balanced_accuracy']*100:.2f}%, Acc: {row['mean_cv_accuracy']*100:.2f}%, C0_Rec: {row['mean_cv_minority_recall']*100:.2f}%, C0_F1: {row['mean_cv_minority_f1']:.4f}, MacroF1: {row['mean_cv_macro_f1']:.4f}, PRAUC: {row['mean_cv_pr_auc']:.4f}, Thresh: {row['threshold']}")
    print(f"   Params: {row['hyperparameters']}")

print("\n=== 4. TOP 10 RANDOM FOREST CANDIDATES BY ACCURACY (Recall0 >= 0.30) ===")
top_acc30 = rf_df[rf_df['mean_cv_minority_recall'] >= 0.30].sort_values(by=['mean_cv_accuracy', 'mean_cv_balanced_accuracy'], ascending=False).head(10)
for idx, (_, row) in enumerate(top_acc30.iterrows(), 1):
    print(f"{idx}. Acc: {row['mean_cv_accuracy']*100:.2f}%, BalAcc: {row['mean_cv_balanced_accuracy']*100:.2f}%, C0_Rec: {row['mean_cv_minority_recall']*100:.2f}%, C0_F1: {row['mean_cv_minority_f1']:.4f}, MacroF1: {row['mean_cv_macro_f1']:.4f}, PRAUC: {row['mean_cv_pr_auc']:.4f}, Thresh: {row['threshold']}")
    print(f"   Params: {row['hyperparameters']}")

print("\n=== 5. TOP 10 RANDOM FOREST CANDIDATES BY ACCURACY (Recall0 >= 0.40) ===")
top_acc40 = rf_df[rf_df['mean_cv_minority_recall'] >= 0.40].sort_values(by=['mean_cv_accuracy', 'mean_cv_balanced_accuracy'], ascending=False).head(10)
for idx, (_, row) in enumerate(top_acc40.iterrows(), 1):
    print(f"{idx}. Acc: {row['mean_cv_accuracy']*100:.2f}%, BalAcc: {row['mean_cv_balanced_accuracy']*100:.2f}%, C0_Rec: {row['mean_cv_minority_recall']*100:.2f}%, C0_F1: {row['mean_cv_minority_f1']:.4f}, MacroF1: {row['mean_cv_macro_f1']:.4f}, PRAUC: {row['mean_cv_pr_auc']:.4f}, Thresh: {row['threshold']}")
    print(f"   Params: {row['hyperparameters']}")

print("\n=== 6. THRESHOLD DYNAMICS AROUND IMPORTANT REGIONS (0.60 to 0.90) ===")
# For champion configuration: n_est=250, depth=3, split=2, leaf=2, mf=0.5, cw=null
champ_thresh = df[
    (df['model_family'] == 'Random_Forest') &
    (df['hyperparameters'].str.contains('250')) &
    (df['hyperparameters'].str.contains('"max_depth": 3')) &
    (df['hyperparameters'].str.contains('"min_samples_split": 2')) &
    (df['hyperparameters'].str.contains('"min_samples_leaf": 2')) &
    (df['hyperparameters'].str.contains('"max_features": 0.5')) &
    (df['hyperparameters'].str.contains('"class_weight": null'))
].sort_values(by='threshold')

print("Thresh | Acc(%) | BalAcc(%) | C0_Rec(%) | C0_F1 | MacroF1 | PR-AUC")
for _, r in champ_thresh.iterrows():
    if r['threshold'] >= 0.50:
        print(f"{r['threshold']:.2f}   | {r['mean_cv_accuracy']*100:.2f} | {r['mean_cv_balanced_accuracy']*100:.2f}   | {r['mean_cv_minority_recall']*100:.2f}     | {r['mean_cv_minority_f1']:.4f} | {r['mean_cv_macro_f1']:.4f}  | {r['mean_cv_pr_auc']:.4f}")

print("\n=== 7. MODEL FAMILY COMPARISON (RF vs XGBoost vs LightGBM) ===")
summary_df = pd.read_csv('results/diagnostics/final_model_selection_summary.csv')
print(summary_df[['model_family', 'threshold', 'mean_cv_accuracy', 'mean_cv_balanced_accuracy', 'mean_cv_minority_recall', 'mean_cv_minority_f1', 'mean_cv_pr_auc', 'mean_cv_macro_f1']].to_string())

print("\n=== 8. OUTER TEST BENCHMARK ===")
with open('results/diagnostics/final_outer_test_evaluation.json') as f:
    outer_data = json.load(f)
print(json.dumps(outer_data, indent=2))
