FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY params.yaml ./
COPY dvc.yaml ./
COPY .dvc ./.dvc
COPY .dvcignore ./

RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

CMD ["uv", "run", "dvc", "repro"]
