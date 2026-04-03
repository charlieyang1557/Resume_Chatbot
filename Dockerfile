# Stage 1: Build React frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.11-slim AS runtime
WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY resume_chatbot/ ./resume_chatbot/
COPY data/ ./data/

# Copy frontend build
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

EXPOSE 8000

# Railway sets $PORT dynamically; fall back to 8000 for local Docker
CMD uvicorn resume_chatbot.webapp:app --host 0.0.0.0 --port ${PORT:-8000}
