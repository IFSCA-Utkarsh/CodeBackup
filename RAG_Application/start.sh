#!/bin/bash

# Navigate to backend and run FastAPI
cd "$(dirname "$0")/backend"
source RAG/bin/activate
concurrently \
  "uvicorn main:app --reload" \
  "cd ../frontend && npm run dev"
