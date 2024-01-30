#!/bin/sh

docker stop neo4j-container
docker rm neo4j-container
#docker rmi neo4j

docker stop ai_search_container
docker rm ai_search_container
docker rmi fast_ai_search