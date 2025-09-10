import boto3
from AmazonS3Utils.constants import AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET

'''
    Upload trained Rl models or to detect the hold data
    
'''

def upload_to_s3(local_path, s3_key):
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"Uploaded {local_path} to s3://{S3_BUCKET}/{s3_key}")
