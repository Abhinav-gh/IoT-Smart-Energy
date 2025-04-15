#!/bin/bash

set -e

# docker stop @(docker ps -a -q --filter "ancestor=flwr_clientapp:0.0.1")
# docker stop supernode-1 supernode-2 serverapp superlink
# docker network rm flwr-network

echo "Step 1: Creating Docker network..."
docker network create --driver bridge flwr-network || echo "Network already exists"

echo "Step 2: Starting SuperLink..."
docker run --rm \
  -p 9091:9091 -p 9092:9092 -p 9093:9093 \
  --network flwr-network \
  --name superlink \
  --detach \
  flwr/superlink:1.17.0 \
  --insecure \
  --isolation \
  process

echo "Step 3: Starting SuperNode-1..."
docker run --rm \
  -p 9094:9094 \
  --network flwr-network \
  --name supernode-1 \
  --detach \
  flwr/supernode:1.17.0 \
  --insecure \
  --superlink superlink:9092 \
  --node-config "partition-id=0 num-partitions=2" \
  --clientappio-api-address 0.0.0.0:9094 \
  --isolation process

echo "Step 3: Starting SuperNode-2..."
docker run --rm \
  -p 9095:9095 \
  --network flwr-network \
  --name supernode-2 \
  --detach \
  flwr/supernode:1.17.0 \
  --insecure \
  --superlink superlink:9092 \
  --node-config "partition-id=1 num-partitions=2" \
  --clientappio-api-address 0.0.0.0:9095 \
  --isolation process

# echo "Step 4: Building ServerApp Docker image..."
# docker build -f serverapp.Dockerfile -t flwr_serverapp:0.0.1 .

echo "Step 4: Starting ServerApp container..."
docker run --rm \
  --network flwr-network \
  --name serverapp \
  --detach \
  flwr_serverapp:0.0.1 \
  --insecure \
  --serverappio-api-address superlink:9091

# echo "Step 5: Building ClientApp Docker image..."
# docker build -f clientapp.Dockerfile -t flwr_clientapp:0.0.1 .

echo "Step 5: Starting ClientApp-1..."
docker run --rm \
  --name flwr-client-1 \
  --network flwr-network \
  --detach \
  flwr_clientapp:0.0.1 \
  --insecure \
  --clientappio-api-address supernode-1:9094

echo "Step 5: Starting ClientApp-2..."
docker run --rm \
  --name flwr-client-2 \
  --network flwr-network \
  --detach \
  flwr_clientapp:0.0.1 \
  --insecure \
  --clientappio-api-address supernode-2:9095

# flwr run ./App local-deployment --stream