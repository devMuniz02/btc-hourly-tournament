# Portfolio Architecture

## Goal
This repo demonstrates how to convert a Python BTC direction classifier into an Azure Machine Learning MLOps project that is ready for recruiter review.

## Azure Services Used
- Azure Machine Learning Workspace for experiments, jobs, model registration, and endpoint management
- Azure ML Compute Cluster for remote training and batch scoring
- Azure ML Environments for reproducible execution
- Azure Blob Storage through Azure ML datastores and job outputs
- Azure ML Model Registry for versioned model promotion
- Azure ML Managed Online Endpoints for real-time inference
- GitHub Actions with Azure login for CI/CD orchestration

## End-to-End Flow
1. GitHub Actions authenticates to Azure with OIDC or a service principal.
2. Azure ML runs a component pipeline that fetches BTC candles, engineers features, trains challenger models, evaluates the winner, registers the model, and writes a batch prediction artifact.
3. The latest registered model is deployed to a managed online endpoint for real-time scoring.
4. Monitoring and job history stay in Azure ML instead of inside Git-tracked runtime artifacts.

## Recruiter Signal
- Reusable componentized pipeline instead of a single script
- Versioned environment and compute definitions
- Real-time deployment template
- Clear CI/CD automation path
- Public docs that explain why each Azure service is present
