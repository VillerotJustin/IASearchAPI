# Use an official Python runtime as a parent image
FROM python:3.11.2-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update
RUN apt-get update && \
    apt-get install -y build-essential


# Install any needed dependencies specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --upgrade pip setuptools
RUN pip install --no-cache-dir -r requirements.txt || { echo "Failed to install dependencies"; exit 1; }

# Define environment variable
ENV NAME World

# RUN Neo4J



# Run app.py when the container launches |uvicorn app.main:app --reload|
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
