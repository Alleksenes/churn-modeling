# ============================================
# File: Dockerfile
# ============================================

# --- Base Stage ---
# Use an official Python runtime as a parent image
# Choose a version compatible with your poetry.lock file (e.g., 3.9, 3.10, 3.11)
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Poetry specific env vars:
ENV POETRY_VERSION=1.7.1 # Use a specific version of Poetry
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true # Create .venv inside project dir (optional)
ENV POETRY_NO_INTERACTION=1 # Disable interactive prompts

# System dependencies (if needed, e.g., for certain ML libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package gcc build-essential && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"
# Add poetry to path
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# --- Builder Stage ---
# Used to install dependencies using Poetry
FROM base as builder

# Copy only files needed for dependency installation
COPY pyproject.toml poetry.lock ./

# Install dependencies
# --no-root: Don't install the project package itself yet
# --no-dev: Exclude development dependencies for the final image
# Use --only main if you don't have dependency groups other than dev
RUN poetry install --no-root --no-dev

# --- Final Stage ---
# Copy installed dependencies and application code
FROM base as final

# Copy the virtual environment with dependencies from the builder stage
COPY --from=builder ${POETRY_HOME} ${POETRY_HOME}
COPY --from=builder /app/.venv /app/.venv

# Copy the application code
# Ensure .dockerignore is set up correctly to avoid copying unnecessary files
COPY ./src ./src
COPY ./config.yaml ./config.yaml
# Copy models directory if models are baked into the image (alternative: mount volume or load from storage)
COPY ./models ./models

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using Uvicorn
# Use the virtual environment's python
CMD ["/app/.venv/bin/uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Healthcheck (optional but recommended)
# HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
#   CMD curl --fail http://localhost:8000/ || exit 1
