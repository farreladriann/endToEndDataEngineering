# docker/airflow/Dockerfile
FROM apache/airflow:2.7.1-python3.9

USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create models directory and set permissions
RUN mkdir -p /opt/airflow/models && \
    chown -R airflow:root /opt/airflow/models && \
    chmod -R 755 /opt/airflow/models

USER airflow

# Copy requirements file
COPY requirements.txt /requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r /requirements.txt

# Copy the entire project
COPY . /opt/airflow/

# Set the PYTHONPATH
ENV PYTHONPATH=/opt/airflow