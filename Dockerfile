FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOST=0.0.0.0

# Install system dependencies (build tools for C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user:user . .

# Ensure upload and db directories exist with write permissions
RUN mkdir -p uploads && chown -R user:user /app

USER user

# Expose Hugging Face default port
EXPOSE 7860

# Start FastAPI server
CMD ["python", "backend/main.py"]
