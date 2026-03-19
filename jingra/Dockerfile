FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code (no datasets; mount or download at runtime)
COPY src/ ./src/
COPY schemas/ ./schemas/
COPY queries/ ./queries/
RUN mkdir -p datasets

# Copy config template (actual config will be mounted via ConfigMap)
COPY jingra.yaml.template ./jingra.yaml.template

ENTRYPOINT ["uv", "run", "jingra"]
