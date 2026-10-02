# Use official Python 3.12 slim image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Declare build arguments passed from CD workflow
ARG MODEL
ARG GROQ_API_KEY
ARG HF_TOKEN

ARG DATABASE_URL
ARG DATABASE_URL_UNPOOLED

ARG AWS_ENDPOINT_URL_S3
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_REGION
ARG S3_BUCKET

ARG PGHOST
ARG PGDATABASE
ARG PGUSER
ARG PGPASSWORD
ARG PGSSLMODE
ARG PGCHANNELBINDING

ARG PORT
# Set them as ENV variables so application process can read them

ENV MODEL=$MODEL
ENV GROQ_API_KEY=$GROQ_API_KEY
ENV HF_TOKEN=$HF_TOKEN

ENV DATABASE_URL=$DATABASE_URL
ENV DATABASE_URL_UNPOOLED=$DATABASE_URL_UNPOOLED

ENV AWS_ENDPOINT_URL_S3=$AWS_ENDPOINT_URL_S3
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_REGION=$AWS_REGION
ENV S3_BUCKET=$S3_BUCKET

ENV PGHOST=$PGHOST
ENV PGDATABASE=$PGDATABASE
ENV PGUSER=$PGUSER
ENV PGPASSWORD=$PGPASSWORD
ENV PGSSLMODE=$PGSSLMODE
ENV PGCHANNELBINDING=$PGCHANNELBINDING
ENV PORT=$PORT

# Install necessary system dependencies for building C extensions / PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv binary from Astral official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project configuration files for layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen --no-cache --no-dev

# Copy application codebase
COPY . .

# Expose FastAPI application port
EXPOSE 8000

# Run Uvicorn server
CMD uvicorn app:app --host 0.0.0.0 --port 8000
