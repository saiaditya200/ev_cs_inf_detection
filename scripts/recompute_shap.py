import pandas as pd

df = pd.read_csv('results/xai/joint_rf_0257/joint_rf_0257_shap_feature_importance.csv')
total = df['mean_abs_shap'].sum()
df['calc_pct'] = (df['mean_abs_shap'] / total) * 100.0

print(f"Total mean_abs_shap: {total:.17f}")
for i, r in df.head(10).iterrows():
    print(f"{i+1:2d}. {r['feature']:<32} : mean_abs_shap = {r['mean_abs_shap']:.17f} | pct = {r['calc_pct']:.2f}% (stored: {r['relative_importance_pct']:.2f}%)")
