FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (as root)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# ─── Create non-root user AFTER all root operations ───
# But keep the files owned by root so we can use curl for health checks
# Actually, let's just run as root for simplicity

# Expose the port
EXPOSE 8080

# Health check using curl (more reliable)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/status || exit 1

# Run the application
CMD ["python", "-u", "bot_server.py"]
