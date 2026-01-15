# Use Python 3.12-slim as the base
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies
# Added "pyyaml" for the config loader
RUN uv pip install --system \
    "boto3>=1.42.26" \
    "langchain-aws>=1.2.0" \
    "langchain-core>=1.2.7" \
    "python-dotenv>=1.0.0" \
    "pyyaml>=6.0"

# Copy all application files (main.py, masking_service.py, monitoring.py, masking_config.yaml)
COPY . . 

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the script
CMD ["python", "main.py"]