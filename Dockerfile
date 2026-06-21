FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    curl tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the API-only dependency set. The full development/Streamlit
# requirements include torch, transformers, and sentence-transformers; those
# are not needed on Render because production embeddings use Voyage's REST API
# and they make the 512 MiB service exceed its memory limit at startup.
COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

# Copy source before editable install
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY pyproject.toml /app/
# NOTE: backend/ was previously never copied into the image at all, even
# though ENTRYPOINT below runs `uvicorn backend.main:app` — any fresh build
# (e.g. on Render, or anyone cloning the repo) would fail at container start
# with ModuleNotFoundError: No module named 'backend'. Fixed by adding this.
COPY backend/ /app/backend/

RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/data/raw_pdfs /app/data/images

# Firebase Admin service account.
# This used to hard-require a firebase-service-account.json file to exist in
# the build context via `COPY firebase-service-account.json /app/` — since
# that file is gitignored (it's a secret), that COPY broke the build on any
# fresh clone (Render, CI, a new teammate) with "file not found". Removed.
#
# Two supported ways to provide it now (see backend/main.py _init_firebase):
#   1. (recommended for Render/Railway/etc.) Set env var
#      FIREBASE_SERVICE_ACCOUNT_JSON to the full contents of the downloaded
#      service-account JSON file — no file needed at all.
#   2. Mount the JSON as a file at runtime (e.g. a Docker secret, Render
#      "secret file", or a local bind mount) and set
#      FIREBASE_SERVICE_ACCOUNT_PATH to that file's path inside the container.

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8000}/api/health" || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
