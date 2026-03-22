# kubera

Name inspired by k8s.

End-to-end ML deployment pipeline. Fine-tuned a Faster R-CNN object detection model on Pascal VOC, served it as a FastAPI REST API, containerized with Docker multi-stage builds, deployed on Kubernetes (Minikube), and automated the full build-test-deploy cycle with GitHub Actions.

## Architecture

```
                         +-------------------+
                         |   model/train.py  |
                         |  Fine-tune Faster |
                         |  R-CNN on VOC     |
                         +---------+---------+
                                   |
                                   v
+----------------+       +-------------------+       +------------------+
|  Client        | POST  |   api/main.py     |       |  mlops/          |
|  (curl/app)    +------>+   FastAPI server   +------>+  metrics.py      |
|                | /detect|  /health, /detect |       |  batch_inference |
+----------------+       +--------+----------+       |  versioning.py   |
                                  |                   +------------------+
                                  v
                         +-------------------+
                         |  model/detect.py  |
                         |  Faster R-CNN     |
                         |  inference        |
                         +-------------------+
```

**Model** -- Faster R-CNN with ResNet-50 FPN backbone. Pretrained on COCO, fine-tuned on Pascal VOC (20 object classes). Model loads once at server startup, runs inference on each request.

**API** -- Two endpoints. `GET /health` returns status and model version. `POST /detect` accepts an image file, runs inference, returns bounding boxes with labels and confidence scores.

**MLOps Utilities** -- Three standalone modules. `metrics.py` computes IoU, precision, and latency benchmarks. `batch_inference.py` processes image directories and outputs JSON reports. `versioning.py` manages a file-based model registry with registration, rollback, and version listing.

**Infrastructure** -- Multi-stage Docker builds for smaller images. Kubernetes Deployment with 2 replicas, health probes, and resource limits. GitHub Actions pipeline running tests, building the image, and deploying to Minikube.

## Tech Stack

- Python 3.11
- PyTorch 2.2 / torchvision 0.17
- FastAPI 0.109
- Pytest 8.0
- Docker (multi-stage builds)
- Kubernetes / Minikube
- OpenCV 4.9 (real-time webcam detection)
- GitHub Actions

## Docker Image Optimization

Multi-stage builds produce a significantly smaller production image by isolating build dependencies from the runtime.

| Build Type   | Virtual Size | Disk Usage |
|-------------|-------------|------------|
| Single-stage | 1.35 GB     | 354 MB     |
| Multi-stage  | 1.05 GB     | 216 MB     |

The multi-stage Dockerfile installs dependencies in a builder stage, then copies only the installed packages into a clean `python:3.11-slim` image. The final image contains no pip cache, no build tools, and no test files.

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=api --cov=model --cov=mlops --cov-report=term-missing

# Start the API
uvicorn api.main:app --reload

# Test the endpoint
curl -X POST "http://localhost:8000/detect" -F "file=@test_image.jpg"
curl http://localhost:8000/health
```

## Real-Time Webcam Detection

```bash
python webcam.py
```

Opens your webcam and runs Faster R-CNN inference on each frame. Detected objects get bounding boxes with labels and confidence scores drawn in real time. Press `q` to quit.

The first frame takes a few seconds while the model loads. After that, inference runs continuously. Works with any USB or built-in webcam.

## Build and Run with Docker

```bash
# Build the image
docker build -t kubera:latest .

# Run the container
docker run -p 8000:8000 kubera:latest

# Test it
curl -X POST "http://localhost:8000/detect" -F "file=@test_image.jpg"
```

## Deploy to Kubernetes (Minikube)

```bash
# Start Minikube
minikube start

# Point Docker CLI to Minikube's daemon
eval $(minikube docker-env)

# Build image inside Minikube
docker build -t kubera:latest .

# Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Wait for rollout
kubectl rollout status deployment/kubera

# Get the service URL
minikube service kubera-service
```

The deployment runs 2 replicas with readiness and liveness probes on `/health`. The service exposes port 30080 via NodePort.

## CI/CD Pipeline

GitHub Actions runs on every push and PR to `main`:

1. **Test** -- Install dependencies, run all tests, enforce 95% coverage threshold
2. **Build** -- Build Docker image tagged with commit SHA (main branch only)
3. **Deploy** -- Set up Minikube, build image, apply K8s manifests, verify rollout (main branch only)

## Project Structure

```
kubera/
├── model/
│   ├── train.py              # Fine-tuning (Faster R-CNN on Pascal VOC)
│   ├── detect.py              # Inference module
│   └── weights/               # Saved model weights (gitignored)
├── api/
│   ├── main.py                # FastAPI endpoints
│   └── schemas.py             # Pydantic response models
├── mlops/
│   ├── metrics.py             # IoU, precision, latency benchmarks
│   ├── batch_inference.py     # Batch processing
│   └── versioning.py          # Model registry
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   ├── test_metrics.py
│   ├── test_batch.py
│   └── test_versioning.py
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── .github/workflows/
│   └── ci-cd.yaml
├── webcam.py                  # Real-time webcam object detection
├── Dockerfile
├── requirements.txt
└── setup.cfg
```

## Test Coverage

42 tests across 5 test files. 96% coverage with `model/train.py` excluded (CLI training script, requires dataset).

```
Name                       Stmts   Miss  Cover
-----------------------------------------------
api/main.py                   21      0   100%
api/schemas.py                15      0   100%
mlops/batch_inference.py      27      2    93%
mlops/metrics.py              29      0   100%
mlops/versioning.py           40      0   100%
model/detect.py               20      4    80%
-----------------------------------------------
TOTAL                        152      6    96%
```
