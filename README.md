# kubera


Name inspired by minikube.
Smart city traffic monitoring and MLOps pipeline. A Faster R-CNN object detection model powers a real-time video processing system that detects vehicles and pedestrians, flags congestion and anomalies, and generates traffic analytics reports. The model is served as a FastAPI REST API, containerized with Docker multi-stage builds, deployed on Kubernetes (Minikube), and automated with GitHub Actions CI/CD.


## Architecture

```
+------------+       +-------------------+       +-------+       +------------------+
| Video      | POST  |   api/main.py     |       |       |       | event_consumer   |
| Stream     +------>+   FastAPI /detect  +------>+ Redis +------>+ Congestion alerts|
| Processor  |       |   Faster R-CNN    |       | Queue |       | Anomaly detection|
+------------+       +-------------------+       +-------+       +--------+---------+
                              |                                            |
                     +--------+----------+                        +--------+---------+
                     |  mlops/           |                        | analytics.py     |
                     |  metrics.py       |                        | Summary reports  |
                     |  batch_inference  |                        | traffic/report   |
                     |  versioning.py    |                        +------------------+
                     +-------------------+
```

**Traffic Monitoring** -- The stream processor reads video frames, POSTs each to the detection API, filters for vehicles and pedestrians, and publishes events to a Redis queue. The consumer reads events, writes JSONL logs, and triggers congestion and anomaly alerts in real time. The analytics module generates summary reports from the event log.

**Model** -- Faster R-CNN with ResNet-50 FPN backbone. Pretrained on COCO, fine-tuned on Pascal VOC (20 object classes). Model loads once at server startup, runs inference on each request.

**API** -- Two endpoints. `GET /health` returns status and model version. `POST /detect` accepts an image file, runs inference, returns bounding boxes with labels and confidence scores.

**MLOps** -- Three standalone modules. `metrics.py` computes IoU, precision, and latency benchmarks. `batch_inference.py` processes image directories and outputs JSON reports. `versioning.py` manages a file-based model registry with registration, rollback, and version listing.

**Infrastructure** -- Multi-stage Docker builds for smaller images. Kubernetes Deployment with 2 replicas, health probes, and resource limits. Docker Compose for the full traffic stack (API + Redis + stream processor). GitHub Actions pipeline running tests, building the image, and deploying to Minikube.

## Tech Stack

- Python 3.11
- PyTorch 2.2 / torchvision 0.17
- FastAPI 0.109
- Pytest 8.0
- Docker (multi-stage builds)
- Kubernetes / Minikube
- OpenCV 4.9 (real-time webcam detection)
- Redis (event queue for traffic pipeline)
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

## Traffic Monitoring

Real-time video stream processing with congestion detection and anomaly alerts.

The stream processor reads video frames, POSTs each to the `/detect` endpoint, and filters for vehicles and pedestrians. Results are published to a Redis queue. A consumer reads the queue, writes events to JSONL, and triggers alerts when vehicle counts exceed a threshold for consecutive frames or detections appear anomalous (low confidence or oversized bounding boxes). The analytics module generates summary reports from the event log.

```bash
# Run with Docker Compose (place video files in ./videos/)
docker compose up

# Run analytics on collected events
python -m traffic.analytics
```

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
│   ├── test_versioning.py
│   └── test_traffic.py
├── traffic/
│   ├── stream_processor.py    # Video → API → Redis producer
│   ├── event_consumer.py      # Redis → JSONL consumer + alerts
│   ├── analytics.py           # Summary report generation
│   └── Dockerfile             # Lightweight image for traffic services
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── traffic-deployment.yaml
├── .github/workflows/
│   └── ci-cd.yaml
├── webcam.py                  # Real-time webcam object detection
├── docker-compose.yaml        # API + Redis + stream processor
├── Dockerfile
├── requirements.txt
└── setup.cfg
```

## Test Coverage

58 tests across 6 test files. Core modules (api, model, mlops) hold 96% coverage with `model/train.py` excluded. Traffic module tests cover all detection logic (congestion, anomaly, filtering, analytics).

```
Name                          Stmts   Miss  Cover
---------------------------------------------------
api/main.py                      21      0   100%
api/schemas.py                   15      0   100%
mlops/batch_inference.py         27      2    93%
mlops/metrics.py                 29      0   100%
mlops/versioning.py              40      0   100%
model/detect.py                  20      4    80%
traffic/analytics.py             45     12    73%
traffic/event_consumer.py        48     26    46%
traffic/stream_processor.py      73     53    27%
---------------------------------------------------
```

Traffic modules have lower coverage because the I/O paths (Redis, HTTP, video capture) are integration-tested via Docker Compose, not unit tests. The pure logic functions (filtering, congestion detection, anomaly detection, report generation) are fully tested.
