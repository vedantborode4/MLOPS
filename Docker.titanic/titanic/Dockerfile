# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the local files (including the Titanic dataset and Python scripts) to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir pandas matplotlib seaborn

# Set the entry point to run the Python script
CMD ["python", "titanic_eda.py"]
