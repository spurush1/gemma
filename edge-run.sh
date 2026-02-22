#!/bin/bash
set -e

echo "=============================================="
echo "ICD-10 MedGemma — Edge Deployment (M4 Mac Mini)"
echo "=============================================="
echo "API-first: all inference via featherless-ai (no local models)"
echo "Total RAM: ~832MB  (well within 16GB)"
echo ""

echo "Building and starting edge services..."
docker-compose -f docker-compose.edge.yml up -d --build

echo ""
echo "Waiting for backend to be healthy..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "  Backend is up!"
    break
  fi
  echo "  Waiting... ($i/30)"
  sleep 3
done

echo ""
echo "=============================================="
echo "  Services running (edge profile):"
echo "  Backend API:        http://localhost:8000"
echo "  Trust Dashboard:    http://localhost:8501"
echo "  HTML Frontend:      http://localhost:80"
echo "=============================================="
echo ""
echo "Memory usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}"
echo ""
echo "View logs:  docker-compose -f docker-compose.edge.yml logs -f"
echo "Stop all:   docker-compose -f docker-compose.edge.yml down"
