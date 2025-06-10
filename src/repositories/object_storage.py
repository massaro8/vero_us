import os
from pathlib import Path
from minio import Minio

MINIO_URL = os.environ.get("MINIO_URL")
MINIO_USERNAME = os.environ.get("MINIO_USERNAME")
MINIO_PASSWORD = os.environ.get("MINIO_PASSWORD")

minio_client = Minio(
            MINIO_URL,
            MINIO_USERNAME,
            MINIO_PASSWORD,
            secure=True
        )