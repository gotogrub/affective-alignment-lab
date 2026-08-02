FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ARG STRUCTURA_GIT_SHA
ARG STRUCTURA_REPOSITORY=gotogrub/affective-alignment-lab

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/artifacts/huggingface \
    TRANSFORMERS_CACHE=/artifacts/huggingface \
    TOKENIZERS_PARALLELISM=false \
    STRUCTURA_IMAGE_GIT_SHA=${STRUCTURA_GIT_SHA} \
    STRUCTURA_REPOSITORY=${STRUCTURA_REPOSITORY}

RUN printf '%s' "$STRUCTURA_GIT_SHA" | grep -Eq '^[0-9a-f]{40}$'
LABEL org.opencontainers.image.source="https://github.com/gotogrub/affective-alignment-lab" \
      org.opencontainers.image.revision="${STRUCTURA_GIT_SHA}"

COPY requirements-structura-qlora.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir -r /tmp/requirements.lock
RUN python -m pip freeze --all > /opt/structura-environment.lock

COPY pyproject.toml README.md ./
COPY dvc.lock structura-dataset-manifest.sha256 ./
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
RUN python -m pip install --no-deps -e .

ENTRYPOINT ["python", "-m", "structura_worker"]
