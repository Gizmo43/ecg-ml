# syntax=docker/dockerfile:1.7
#
# Image for training / evaluation on NVIDIA GPUs.
#
# Build (from the project root):
#   docker build -f docker/Dockerfile -t classification:gpu .
#
# Run (requires the NVIDIA driver >= 580 and NVIDIA Container Toolkit on the host):
#   docker run --rm --gpus all --ipc=host \
#     -v "$(pwd)/data:/app/data" -v "$(pwd)/logs:/app/logs" \
#     --env-file .env \
#     classification:gpu python -m ecg_ml.train experiment=test trainer.accelerator=gpu

# torch 2.14.1 wheels bundle the CUDA 13.0 runtime, cuDNN and NCCL, so the slim
# `base` flavour (driver plumbing + NVIDIA_* env vars) is enough
ARG CUDA_VERSION=13.0.3
ARG UBUNTU_VERSION=24.04
ARG PYTHON_VERSION=3.13

# --------- builder: resolve the locked environment with uv --------- #
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION} AS builder
ARG PYTHON_VERSION

COPY --from=ghcr.io/astral-sh/uv:0.9.13 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON=${PYTHON_VERSION}

RUN uv python install ${PYTHON_VERSION}

WORKDIR /app

# dependencies first, so code changes don't invalidate this heavy layer
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-dev --no-install-project

COPY pyproject.toml ./
COPY config ./config
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

# --------- runtime --------- #
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

# the venv's interpreter symlinks into /python, so both keep their builder paths
COPY --from=builder /python /python
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# data/ and logs/ are mounted at runtime (paths.data_dir / paths.log_dir)
VOLUME ["/app/data", "/app/logs"]

# trainer/default.yaml targets mps, so pass trainer.accelerator=gpu when overriding
CMD ["python", "-m", "ecg_ml.train", "experiment=test", "trainer.accelerator=gpu"]
