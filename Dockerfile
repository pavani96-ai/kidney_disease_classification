# Use a slim image, but be aware you may need to install NVIDIA drivers 
# in your runtime environment (like Azure Machine Learning or AKS)
FROM python:3.11-slim-bookworm

# Install basic OS dependencies
RUN apt update -y && apt install -y curl gpg lsb-release && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Copy ONLY the requirements file first
# This allows Docker to cache the installation layer
COPY Requirements.txt .

# Step 2: Install dependencies
RUN pip install --no-cache-dir -r Requirements.txt

# Step 3: Copy the rest of the application
COPY . .

# Set environment variables for better performance
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python3", "app.py"]