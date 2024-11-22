#!/bin/bash

# Stop containers
docker-compose down

# Remove existing volumes
docker volume rm endtoenddataengineering_postgres-data

# Create directories with correct permissions
sudo mkdir -p ./airflow/logs ./airflow/dags ./airflow/plugins
sudo chown -R 50000:50000 ./airflow/logs ./airflow/dags ./airflow/plugins

# Start containers
docker-compose up -d

# Initialize the database
docker-compose exec airflow-webserver airflow db init

# Create admin user
docker-compose exec airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin