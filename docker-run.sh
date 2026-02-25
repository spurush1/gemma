#!/bin/bash
set -e

echo "======================================"
echo "ICD-10 MedGemma — Full Stack Startup"
echo "======================================"

echo ""
echo "Building images..."
docker-compose build

echo ""
echo "Starting all services (Neo4j + Backend + Frontend + Streamlit)..."
docker-compose up -d

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
echo "======================================"
echo "  Services running:"
echo "  Backend API:        http://localhost:8000"
echo "  API docs (Swagger): http://localhost:8000/docs"
echo "  Trust Dashboard:    http://localhost:9501"
echo "  HTML Frontend:      http://localhost:80"
echo "  Neo4j Browser:      http://localhost:7476  (neo4j / medgemma123)"
echo "======================================"
echo ""
echo "View logs:      docker-compose logs -f"
echo "Stop all:       ./docker-stop.sh"
echo "Run validation: python tests/validate_100_cases.py"
