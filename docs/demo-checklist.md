# Demo Checklist

## Fast recruiter walkthrough
1. Open the repo root README.
2. Review the Azure architecture diagram and service map.
3. Inspect `azureml/pipelines/train_register_score.yml`.
4. Inspect `.github/workflows/train-azure.yml` and `.github/workflows/deploy-endpoint.yml`.
5. Review `azureml/endpoints/` for the real-time serving story.

## Commands to show in a live demo
```bash
az ml compute create --file azureml/compute/cpu-cluster.yml
az ml environment create --file azureml/environments/training-env.yml
az ml job create --file azureml/pipelines/train_register_score.yml --set inputs.registered_model_name=btc-direction-model
az ml online-endpoint create --file azureml/endpoints/btc-direction-endpoint.yml
az ml online-deployment create --file azureml/endpoints/btc-direction-deployment.yml --all-traffic
az ml online-endpoint invoke -n btc-direction-endpoint --request-file azureml/sample-request.json
```

## What to emphasize
- The model code was reused, not replaced.
- The repo demonstrates packaging, orchestration, deployment, and CI/CD.
- Azure ML became the operating platform, while MLflow stayed as the lifecycle layer inside Azure.
