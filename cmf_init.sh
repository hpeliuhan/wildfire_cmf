export CMF_SERVER_IP=
export MINIO_IP=
export MINIO_USER=
export MINIO_PASSWORD=
export GIT_URL=
export NEO4J_SERVER_IP=
export NEO4J_USER=
export NEO4J_PASSWORD=
cmf init minioS3 --url s3://art  --endpoint-url http://${MINIO_IP}:9000 --access-key-id $MINIO_USER --secret-key $MINIO_PASSWORD --git-remote-url $GIT_URL  --cmf-server-url http://${CMF_SERVER_IP}:8080 --neo4j-user $NEO4J_USER --neo4j-password $NEO4J_PASSWORD --neo4j-uri bolt://${NEO4J_SERVER_IP}:7687
