FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/artifacts/huggingface \
    TRANSFORMERS_CACHE=/artifacts/huggingface \
    TOKENIZERS_PARALLELISM=false

COPY requirements-structura-qlora.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir -r /tmp/requirements.lock
RUN python -m pip freeze --all > /opt/structura-environment.lock

COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
RUN python -m pip install --no-deps -e .

ENTRYPOINT ["python"]
