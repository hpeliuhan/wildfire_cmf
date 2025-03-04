export CMF_SERVER_IP=192.168.30.116
export MINIO_IP=192.168.30.116
export MINIO_USER=minioadmin
export MINIO_PASSWORD=minioadmin
export GIT_URL=https://github.com/hpeliuhan/wildfire_cmf.git
export NEO4J_SERVER_IP=192.168.30.116
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=test1234
cmf init minioS3 --url s3://dvc-art  --endpoint-url http://${MINIO_IP}:9000 --access-key-id $MINIO_USER --secret-key $MINIO_PASSWORD --git-remote-url $GIT_URL  --cmf-server-url http://${CMF_SERVER_IP}:8080 --neo4j-user $NEO4J_USER --neo4j-password $NEO4J_PASSWORD --neo4j-uri bolt://${NEO4J_SERVER_IP}:7687
