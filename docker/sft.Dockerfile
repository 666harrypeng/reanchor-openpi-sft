# Training image: CUDA 12 runtime + uv; the openpi environment is built from the submodule's lock.
#   docker build -f docker/sft.Dockerfile -t reanchor-openpi-sft .
#   docker run --rm -it --gpus all --ipc=host -e HF_TOKEN -e WANDB_API_KEY \
#     -v "$PWD:/workspace" -v /data/reanchor-sft:/workspace/.cache reanchor-openpi-sft bash
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python PYTHONUNBUFFERED=1 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
WORKDIR /workspace
COPY . /workspace
RUN uv python install 3.11 && bash tools/openpi/sync.sh
ENV PATH=/workspace/tools/openpi/.venv/bin:$PATH
