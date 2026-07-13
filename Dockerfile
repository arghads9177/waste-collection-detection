FROM python:3.11-slim

WORKDIR /app

# System deps commonly needed by CV stacks (opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements/requirements.txt requirements/requirements.txt
RUN pip install --no-cache-dir -r requirements/requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY configs/.env.example .env

ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["python", "scripts/infer.py"]
