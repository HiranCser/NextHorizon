# ---------- Stage 1: Build base dependencies ------
FROM python:3.12-slim AS builder

# Disable bytecode and buffering for cleaner logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for numpy, pandas, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file
COPY requirements.txt .

# Install Python dependencies in a virtual environment (optional)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: Final runtime image ----------
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder image
COPY --from=builder /usr/local /usr/local

# Copy the rest of the application
COPY . /app

# Expose Streamlit port
EXPOSE 8080

# Environment variables (you can override at runtime)
ENV STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Default startup command
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]

