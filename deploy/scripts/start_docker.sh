#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aaws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 650251684767.dkr.ecr.us-east-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 650251684767.dkr.ecr.us-east-1.amazonaws.com/delivery_time:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=delivery_time)" ]; then
    echo "Stopping existing container..."
    docker stop delivery_time
fi

if [ "$(docker ps -aq -f name=delivery_time)" ]; then
    echo "Removing existing container..."
    docker rm delivery_time
fi

echo "Starting new container..."
docker run -d -p 80:8000 --name delivery_time -e DAGSHUB_TOKEN=a131814566b9dd9187d50514caafd7f751b63df4 650251684767.dkr.ecr.us-east-1.amazonaws.com/delivery_time:latest

echo "Container started successfully."