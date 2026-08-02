FROM python:3.12-slim

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1

COPY requirements-structura-data.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir -r /tmp/requirements.lock

COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
RUN python -m pip install --no-deps -e .

ENTRYPOINT ["dvc"]
