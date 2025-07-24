# ============================================
# File: Dockerfile
# ============================================

# --- Base Stage ---
# Use an official Python runtime as a parent image
# Choose a version compatible with your poetry.lock file (e.g., 3.9, 3.10, 3.11)
FROM python:3.9-slim as base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Poetry specific env vars:
ENV POETRY_VERSION=1.7.1 # Use a specific version of Poetry
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true # Create .venv inside project dir (optional)
ENV POETRY_NO_INTERACTION=1 # Disable interactive prompts

RUN pip install "poetry==$POETRY_VERSION"
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

# --- Builder Stage ---
FROM base as builder

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root --no-dev

# --- Final Stage ---
FROM base as final

COPY --from=builder ${POETRY_HOME} ${POETRY_HOME}
COPY --from=builder /app/.venv /app/.venv

COPY ./src ./src
COPY ./config.yaml ./config.yaml
COPY ./models ./models

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
