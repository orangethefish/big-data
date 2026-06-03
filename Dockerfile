# syntax=docker/dockerfile:1.7

FROM python:3.10-slim-bookworm

ARG TARGETPLATFORM
ARG BUILDPLATFORM

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HARM_DETECTION_ROOT=/workspace \
    JAVA_HOME=/usr/lib/jvm/default-java \
    SPARK_LOCAL_IP=127.0.0.1

WORKDIR /opt/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        default-jre-headless \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY docker ./docker

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[dev]"

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash app \
    && mkdir -p /workspace \
    && chown -R app:app /opt/app /workspace

USER app
WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini", "--", "/bin/sh", "/opt/app/docker/entrypoint.sh"]
CMD ["harm-detect", "--help"]
