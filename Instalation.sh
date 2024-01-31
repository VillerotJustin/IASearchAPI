#!/bin/sh

# Programme variables
username=$1
password=$2
prefix=$3
app_password=$4
secret_key=$5

# Check if .env exists, and if so, delete it
if [ -f .env ]; then
    rm .env
fi

# Run Neo4J Docker
docker stop neo4j-container
docker rm neo4j-container
docker volume rm neo4j_data
docker volume create neo4j_data
# Check if .env exists, and if so, delete it
if [ -d ../neo4j_data ]; then
    rm -r ../neo4j_data
fi
mkdir ../neo4j_data
mkdir neo4j_plugins
printf "\n./neo4j_plugins/\n" >> .dockerignore
cd neo4j_plugins
wget "https://github.com/neo4j-labs/neosemantics/releases/download/5.15.0/neosemantics-5.15.0.jar"
wget "https://github.com/neo4j/graph-data-science/releases/download/2.5.7/neo4j-graph-data-science-2.5.7.jar"
cd ..
docker run -d --name neo4j-container -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=$username/$password -v neo4j_data:$(readlink -f ../neo4j_data) --volume=$HOME/neo4j/plugins:$(readlink -f ./neo4j_plugins) neo4j:latest

# Use a subshell to execute docker inspect
ip_neo4j=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' neo4j-container)
#echo "$1 + $2 + $3 + $4 + $5 + $ip_neo4j"

# Create .env
touch .env

printf "APP_NAME=\"IA Search API\"\n" >> .env
printf "APP_DESC=\"API built for IA enhanced research on Neo4j with FastAPI\"\n"  >> .env
printf "APP_VERSION=\"0.1\"\n"  >> .env
printf "DOCS_URL=\"/docs\"\n"  >> .env
printf "REDOC_URL=\"/redoc\"\n"  >> .env
printf "NEO4J_URI=\"bolt://$ip_neo4j:7687\"\n"  >> .env
printf "NEO4J_USERNAME=\"$username\"\n"  >> .env
printf "NEO4J_PASSWORD=\"$password\"\n"  >> .env
printf "DB_PREFIX=\"$prefix\"\n"  >> .env
printf "APP_PASSWORD=\"$app_password\"\n"  >> .env
printf "SECRET_KEY=\"$secret_key\"\n"  >> .env
printf "ALGORITHM=\"HS256\"\n"  >> .env  # Corrected typo
printf "ACCESS_TOKEN_EXPIRE_MINUTES=10080" >> .env



# Build API Docker
docker build -t fast_ai_search .

# Run API Docker
docker run -d --name ai_search_container -p 9000:80 fast_ai_search