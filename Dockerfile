FROM python:3.11-slim
WORKDIR /app

# Install system libs needed by scikit-image / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

# Install CPU-only PyTorch first (separate index so other packages use PyPI)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install everything else
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY frontend/ ./frontend/
COPY data/ ./data/
RUN mkdir -p /app/logs /app/models

# Pre-download CheXNet weights at build time so first request is fast
# Falls back silently if network unavailable during build
RUN python -c "\
import torchxrayvision as xrv; \
m = xrv.models.DenseNet(weights='densenet121-res224-nih'); \
print('CheXNet weights cached.')" || echo "CheXNet weight pre-download skipped"

EXPOSE 8000 9501
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
