#!/bin/bash

function start_services() {
    echo "Starting services..."
    docker-compose up -d
    echo "Waiting for services to be ready..."
    sleep 10
}

function stop_services() {
    echo "Stopping services..."
    docker-compose down
}

function init_airflow() {
    echo "Initializing Airflow..."
    docker-compose exec airflow-webserver airflow db init
    
    echo "Creating Airflow admin user..."
    docker-compose exec airflow-webserver airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin
}

function show_logs() {
    docker-compose logs -f
}

case "$1" in
    "start")
        start_services
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        start_services
        ;;
    "init")
        init_airflow
        ;;
    "logs")
        show_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|init|logs}"
        exit 1
        ;;
esac