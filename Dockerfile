# Dockerfile
FROM --platform=linux/amd64 python:3.10

WORKDIR /app

# Copy all files in the app directory to the /app directory inside the container
COPY app/ .

# Copy .env file to the container
COPY .env .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the main script
CMD ["python", "main.py"]