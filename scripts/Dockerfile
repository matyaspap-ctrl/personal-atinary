# Use Python 3.12-slim as the base
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install the dependencies explicitly into the system Python.
# We skip 'uv sync' because it struggles with your specific package-mode=false setup.
RUN uv pip install --system \
    "boto3>=1.42.26" \
    "langchain-aws>=1.2.0" \
    "langchain-core>=1.2.7" \
    "python-dotenv>=1.0.0"

# Copy the application code
COPY main.py .
# (Optional) Copy other files if you have them, or just use . .
# COPY . . 

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the script
CMD ["python", "main.py"]
