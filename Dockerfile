FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=bulk_app.py \
    DATA_DIR=/app/data

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN mkdir -p /app/data

EXPOSE 6061

CMD ["gunicorn", "--bind", "0.0.0.0:6061", "--workers", "2", "--timeout", "120", "bulk_app:app"]