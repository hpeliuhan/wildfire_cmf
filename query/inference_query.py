from cmflib import cmfquery
import pandas as pd


# To show all rows and columns without truncation
pd.set_option('display.max_rows', None)    # No limit to rows
pd.set_option('display.max_columns', None) # No limit to columns
pd.set_option('display.width', None)       # Unlimited width
pd.set_option('display.max_colwidth', None)# No limit to column width

import os
import sys

#sys.setrecursionlimit(2000)  # Example to increase recursion limit, helpful in deep lists

from minio import Minio
from urllib.parse import urlparse


def strip_prefix(text, prefix):
    """Remove a specific prefix from a string."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def download_from_minio_s3_url(uri, download_path, access_key, secret_key, endpoint):
    """
    Download a file from MinIO using an S3-style URL.

    Args:
        uri (str): S3-style URI (e.g., "s3://my-bucket/my-folder/my-file.txt").
        download_path (str): Local path to save the downloaded file.
        access_key (str): MinIO access key.
        secret_key (str): MinIO secret key.
        endpoint (str): MinIO server endpoint (e.g., "localhost:9000" or "minio.example.com").
    """
    parsed_url = urlparse(uri)
    
    # Extract bucket and object path
    bucket_name = parsed_url.netloc
    object_name = parsed_url.path.lstrip('/')
    print(bucket_name)
    print(object_name)

    # Initialize MinIO client
    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False  # Set to True if using HTTPS
    )

    # Download the object
    client.fget_object(bucket_name, object_name, download_path)
    print(f"File downloaded successfully to: {download_path}")

# Example usage





#from ..training import utils
#the inference will query the latest model metadata 
#then download the model from the minio server using the metadata
#
pipeline_name="WILDFIRE"
pipeline_file="cmf"

query = cmfquery.CmfQuery(pipeline_file)
pipelines = query.get_pipeline_names()
stages = query.get_pipeline_stages(pipelines[0])
executions= query.get_all_executions_in_stage('WILDFIRE/model_convert')
#print(executions.columns)
latest_execution=executions[executions["id"] == executions["id"].max()].to_dict(orient="records")[0]
#print(latest_execution['id'])

#get artifacts:
convert_artifacts=query.get_all_artifacts_for_execution(latest_execution['id'])
model_artifact=convert_artifacts[convert_artifacts["event"]=="OUTPUT"]
print(model_artifact['url'].to_string(index=False))
s3_link=strip_prefix(model_artifact['url'].to_string(index=False), pipeline_name+":")
#DOWNLOAD ARTIFAFTs:
chunk_size = 100  # Adjust chunk size
for i in range(0, len(s3_link), chunk_size):
    print(s3_link[i:i + chunk_size])
print(s3_link)
model_path="../inferencing/model.tflite"

download_from_minio_s3_url(
    s3_link,
    model_path,
    access_key="minioadmin",
    secret_key="minioadmin",
    endpoint="192.168.30.116:9000"  # Or your MinIO server address
)
