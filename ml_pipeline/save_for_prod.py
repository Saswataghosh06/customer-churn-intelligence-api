# ml_pipeline/save_for_prod.py
import mlflow
import mlflow.sklearn
import joblib

mlflow.set_tracking_uri("http://localhost:5000")
runs = mlflow.search_runs(experiment_names=["churn_prediction_experiment"], max_results=1)
run_id = runs.iloc[0].run_id

print(f"Loading model from run {run_id}...")
model = mlflow.sklearn.load_model(f"runs:/{run_id}/churn_random_forest")

print("Saving to app/ml_models/model.pkl...")
joblib.dump(model, 'app/ml_models/model.pkl')
print("✅ Model baked and ready for production!")