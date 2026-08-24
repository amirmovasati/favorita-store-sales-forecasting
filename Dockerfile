# Base image: a slim Python 3.11 environment (small download, has
# everything pandas/xgboost/fastapi need).
FROM python:3.11-slim

# All commands below run inside this directory in the container.
WORKDIR /app

# Copy only the dependency list first (not the whole project). Docker
# caches this layer, so re-building after a code-only change skips
# re-installing dependencies -- much faster iteration.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual project.
COPY src/ ./src/
COPY run_pipeline.py .
COPY api.py .

# The trained model/artifacts and data are NOT baked into the image
# (they're large and change independently of the code) -- they are
# mounted in at run time instead. See README for the run command.

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
