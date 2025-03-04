from minio import Minio
from minio.error import S3Error

def upload_to_minio(file_path, bucket_name, object_name, minio_client):
    try:
        # Make the bucket if it doesn't exist
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
        
        # Upload the file
        minio_client.fput_object(bucket_name, object_name, file_path)
        print(f"File {file_path} uploaded successfully to bucket {bucket_name} with object name {object_name}")
        return bucket_name
    except S3Error as e:
        print(f"Error: Unable to upload file to MinIO")
        print(e)
        return None

# Example usage
if __name__ == "__main__":
    # Initialize MinIO client
    minio_client = Minio(
        "192.168.30.115:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
    )

    # Upload a file
    file_path = "/data/cmf_sage/wildfire/wildfire_cmf/artifacts/model_convert/convert/model.tflite"
    bucket_name = "waggle"
    object_name = "model.tflite"
    upload_to_minio(file_path, bucket_name, object_name, minio_client)