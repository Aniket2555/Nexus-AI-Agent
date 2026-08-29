FROM python:3.11-slim

# tesseract-ocr: required by pytesseract (image_processor.py). libpq: required by
# psycopg's binary wheel at runtime on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY backend ./backend

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "backend.app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
