# Dockerfile
# FROM --platform=linux/amd64 
FROM python:3.10

# Set the working directory to /app
RUN mkdir /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy dataset to the container
COPY data/ .

# Copy all .py files in the app directory to the /app directory inside the container
COPY app/ .

# Copy .env file to the container
COPY .env .

# Run the main script
CMD ["python", "main.py"]