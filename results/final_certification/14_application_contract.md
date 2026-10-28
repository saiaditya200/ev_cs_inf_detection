# Application & Inference Contract

- **Audited Modules**: `08_predict.py` and `app.py`.
- **Verified Contract Checkpoints**:
  1. Both load `final_deployment_model.joblib`.
  2. Both load `0.76` threshold from `final_deployment_threshold.json`.
  3. Both enforce the certified 21-predictor contract in identical order.
  4. Zero scaling is loaded or applied in the Random Forest inference path.
  5. Target fields (`fast_charging_flag`, `target`, `label`) are stripped before inference.
  6. Equipment specifications (`charging_power_kw`, `charger_type`) are stripped before inference.
  7. Existing station labels are kept separate and descriptive only.
  8. Prospective site inference operates purely on user-specified spatial coordinates and road attributes.
  9. TreeSHAP utilizes the exact identical 21-feature inference vector.
  10. Counterfactuals perturb only the 4 approved planning levers.
  11. District EIPS context is merged downstream without affecting station ML probabilities.
  12. API endpoints and web interface adhere to the identical schema.
