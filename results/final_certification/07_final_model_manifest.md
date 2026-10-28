# Final Model Manifest

The project maintains explicit separation across 5 distinct model identities:

1. **ACTIVE PRODUCTION MODEL**:
   - Runtime: `RandomForestClassifier(n_estimators=250, max_depth=3, min_samples_split=2, min_samples_leaf=2, max_features=0.5, random_state=42)`
   - Decision Threshold: `0.760`
   - Role: Active frozen production deployment artifact (`models/final_deployment_model.joblib`).
   - Production Status: CURRENT_PRODUCTION (Untouched and Frozen).

2. **CURRENT BAKEOFF BENCHMARK**:
   - Runtime: `RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_split=7, min_samples_leaf=1, max_features=0.5, random_state=42)`
   - Decision Threshold: `0.755`
   - Role: Primary bake-off benchmark from Script 09.
   - Production Status: BENCHMARK_ONLY.

3. **JOINT_RF_0257 CANDIDATE**:
   - Runtime: `RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_split=7, min_samples_leaf=1, max_features=0.5, random_state=42)`
   - Decision Threshold: `0.760`
   - Role: Joint optimization candidate from Script 11.
   - Production Status: PREPARED_SEPARATELY.

4. **RF_0002 PARETO BENCHMARK**:
   - Runtime: `RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_split=4, min_samples_leaf=1, max_features=0.5, max_samples=0.7, random_state=42)`
   - Decision Threshold: `0.740`
   - Role: Pareto Band 2 leader from Script 13.
   - Production Status: BENCHMARK_ONLY.

5. **RF_0039 PARETO BENCHMARK**:
   - Runtime: `RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_split=4, min_samples_leaf=1, max_features=0.7, max_samples=None, random_state=42)`
   - Decision Threshold: `0.720`
   - Role: Pareto Band 4 leader from Script 13.
   - Production Status: BENCHMARK_ONLY.
