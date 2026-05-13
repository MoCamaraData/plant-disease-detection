# DEPRECATED: project no longer deployed (GCP free trial expired). Kept for reference.

# Project Journal: FastAPI Service for Plant Disease Detection

## Project goal

The goal of this project was to build an API capable of receiving a leaf image and
automatically identifying the corresponding disease. The API sat between the trained
model (ResNet9) and an external user or application.

It was responsible for:
- loading the saved checkpoint at startup,
- preprocessing incoming images,
- returning a JSON prediction,
- being easy to ship with Docker.

---

## Stack

- FastAPI (API framework)
- Uvicorn (ASGI server)
- PyTorch & Torchvision (model loading and preprocessing)
- Pillow (image I/O)
- python-multipart (file uploads)
- Docker (containerization)
- Python 3.12

## API architecture

Three endpoints:

| Endpoint   | Method | Description                                                |
| ---------- | ------ | ---------------------------------------------------------- |
| `/`        | GET    | Landing page with links                                    |
| `/health`  | GET    | Reports whether the model is loaded                        |
| `/predict` | POST   | Accepts an image and returns the predicted disease class   |

The model is loaded once at startup via the `@app.on_event("startup")` hook.

---

## Loading the model

The saved checkpoint (`plant-disease-model.pth`) contains a `state_dict`. To load it,
the API has to:

1. rebuild the ResNet9 architecture,
2. load the weights with `load_state_dict`,
3. switch to evaluation mode via `model.eval()`.

The model path is:

```python
MODEL_PATH = "./plant-disease-model.pth"
```

## Containerization

Containerization made the API:

- portable across machines,
- independent of the host's Python and package versions,
- runnable with a single Docker command.

The API and the model were bundled into a Docker image based on Python 3.12.

### Image

A custom `plant-api` image was built from a `Dockerfile` that:

- starts from `python:3.12-slim`,
- installs the system dependencies needed by Pillow,
- installs the CPU build of PyTorch,
- installs the API dependencies from `requirements.txt`,
- copies the `plant-disease-model.pth` checkpoint,
- exposes port `8000`,
- starts the Uvicorn server.

The build command was:

```bash
docker build -t plant-api .
```
