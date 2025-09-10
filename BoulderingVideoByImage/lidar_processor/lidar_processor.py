import boto3
from AmazonS3Utils.constants import AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET

'''
    Download files from S3
'''

def download_from_s3(s3_key, local_path):
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
    s3.download_file(S3_BUCKET, s3_key, local_path)
    print(f"Downloaded {s3_key} to {local_path}")