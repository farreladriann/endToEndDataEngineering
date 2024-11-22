#!/bin/bash

# Create project structure
mkdir -p docker/{airflow,jupyter,postgres}
mkdir -p airflow/{dags,logs,plugins,config}
mkdir -p src/{data,database,models,utils}
mkdir -p {notebooks,tests,logs}

# Create necessary files
touch docker/airflow/Dockerfile
touch docker/jupyter/Dockerfile
touch docker/postgres/init.sql
touch .env
touch requirements.txt

# Create __init__.py files
for dir in src src/data src/database src/models src/utils tests; do
    echo 'from src.utils.logger import setup_logger

logger = setup_logger(__name__)' > ${dir}/__init__.py
done

# Set permissions
chmod -R 755 docker/
chmod -R 755 airflow/
chmod -R 755 src/
chmod -R 755 tests/
chmod -R 766 logs/
chmod 644 .env
chmod 644 requirements.txt

echo "Project structure created successfully!"