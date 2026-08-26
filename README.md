# MLOps PyTorch Pipeline

An end-to-end CIFAR-10 image-classification workflow using PyTorch, Docker, and Kubernetes. The project trains a CNN, saves a checkpoint, and serves predictions through a FastAPI API.

## Architecture

```mermaid
flowchart LR
    A[CIFAR-10 data] --> B[PyTorch training job]
    C[training_config.yaml / ConfigMap] --> B
    B --> D[(Checkpoint PVC)]
    D --> E[FastAPI serving Deployment: 2 replicas]
    E --> F[ClusterIP Service: port 80 to 8080]
    F --> G[/health and /predict]
    H[HPA: 2 to 5 replicas] --> E
```

## Repository layout

```text
src/             Model, dataset, training, and FastAPI application
configs/         Local training configuration
docker/          Training and serving Dockerfiles
k8s/             Kubernetes manifests
requirements/    Pinned training and serving dependencies
tests/           Model tests
```

## Prerequisites

- Python 3.11
- Docker Desktop with Kubernetes enabled
- `kubectl` configured for the local cluster

## Local setup and tests

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements\train.txt
python -m pip install -r requirements\serve.txt
python -m pytest -q
```

## Train and serve locally

Train the model. CIFAR-10 is downloaded automatically to `data/`, and the checkpoint is written to `checkpoints/classifier_v1.pt`.

```powershell
python -m src.train
```

In another terminal, start the API:

```powershell
python -m src.serve
```

Check that the checkpoint loaded:

```powershell
curl.exe http://127.0.0.1:8080/health
```

Send an image for prediction:

```powershell
curl.exe -X POST "http://127.0.0.1:8080/predict" -F "image=@test_image.png"
```

## Docker

From the repository root:

```powershell
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

Run training with local data and checkpoint directories mounted into the container:

```powershell
$project = (Get-Location).Path
docker run --rm `
  --mount "type=bind,source=$project\data,target=/app/data" `
  --mount "type=bind,source=$project\checkpoints,target=/app/checkpoints" `
  mlops-train:v1
```

Run the serving image and test it in a second terminal:

```powershell
docker run --rm -p 8080:8080 `
  --mount "type=bind,source=$project\checkpoints,target=/app/checkpoints" `
  mlops-serve:v1

curl.exe http://127.0.0.1:8080/health
curl.exe -X POST "http://127.0.0.1:8080/predict" -F "image=@test_image.png"
```

## Kubernetes deployment

The manifests use the `ml-training` namespace, a ConfigMap for the training configuration, PVCs for data and model checkpoints, a CPU training Job, an optional GPU training Job, and a two-replica serving Deployment.

Build and make the images available to your cluster before applying the manifests. The image names in `k8s/training-job.yaml` and `k8s/serving-deployment.yaml` must match images that the cluster can pull.

For Minikube with the Docker runtime, build the CPU images directly in Minikube's Docker daemon. This avoids transferring large PyTorch image archives with `minikube image load`:

```bash
eval "$(minikube docker-env)"
docker build -f docker/Dockerfile.train -t chathuranj/mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t chathuranj/mlops-serve:v1 .
docker image ls | grep -E 'mlops-train|mlops-serve'
eval "$(minikube docker-env -u)"
```

```powershell
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
kubectl wait --for=condition=complete job/pytorch-training -n ml-training --timeout=20m

kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training
```

Port-forward the service in one terminal:

```powershell
kubectl port-forward svc/model-serving 8080:80 -n ml-training
```

Then test it from a second terminal:

```powershell
curl.exe http://127.0.0.1:8080/health
curl.exe -X POST "http://127.0.0.1:8080/predict" -F "image=@test_image.png"
```

## GPU training bonus

`k8s/training-job-gpu.yaml` is separate from the mandatory CPU Job so the full workflow remains runnable on clusters without a GPU. The GPU Job requests and limits one `nvidia.com/gpu` resource and targets a node labelled `accelerator=nvidia-gpu`.

Build the CUDA 12.4 PyTorch training image from an Ubuntu/WSL2 terminal and load it into Minikube:

```bash
docker build -f docker/Dockerfile.train.gpu -t chathuranj/mlops-train:gpu .
docker run --rm --gpus=all --entrypoint python chathuranj/mlops-train:gpu \
  -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
minikube image load chathuranj/mlops-train:gpu
```

The Kubernetes node must advertise at least one allocatable GPU before applying the Job:

```bash
minikube kubectl -- get nodes \
  -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
minikube kubectl -- label node minikube accelerator=nvidia-gpu --overwrite
minikube kubectl -- apply -f k8s/training-job-gpu.yaml
minikube kubectl -- logs -f job/pytorch-training-gpu -n ml-training
```

Successful GPU training logs `"device": "cuda"` in the structured `training_started` event. Docker Desktop on WSL2 can expose the GPU to Docker and to the Minikube node while still failing to advertise `nvidia.com/gpu` because the nested NVIDIA device plugin reports `Failed to initialize NVML: Not Supported`. In that environment, use `k8s/training-job.yaml` for end-to-end CPU validation and retain the GPU manifest as the production GPU-node configuration.

## API

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Returns HTTP 200 when the checkpoint is loaded. |
| `/predict` | `POST` | Accepts an `image` multipart form field and returns the predicted CIFAR-10 class and probabilities. |

## CI

GitHub Actions runs the test suite for pushes and pull requests targeting `main` or `develop`.
