# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Set environment variables (DATA_DIR so persistent data can be mounted at /app/data)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    DATA_DIR=/app/data

# Copy requirements file
COPY requirements.txt .

# --- ADDED SECTION START ---
# Install build dependencies to compile NumPy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential gcc \
    && rm -rf /var/lib/apt/lists/*
# --- ADDED SECTION END ---

# Copy application code
COPY . .

# Create directory for data files
RUN mkdir -p /app/data

# Expose port
EXPOSE 6060

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:6060", "--workers", "4", "--timeout", "120", "wsgi:application"]