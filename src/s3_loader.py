import os
from pathlib import Path
from typing import List, Optional, TypedDict
import boto3
from dotenv import load_dotenv
from langchain_core.documents import Document

from src.data_loader import DataLoader

from botocore.client import Config

load_dotenv()

class FileMetadata(TypedDict):
    local_path: Path
    name: str
    type: str
    download_url: str
    key: str

FOLDER_MAPPING = {
    ".pdf": "pdfs",
    ".csv": "csvs",
    ".xlsx": "xlsxs",
    ".xls": "xlsxs",
    ".docx": "word",
    ".doc": "word",
    ".txt": "txts",
    ".json": "jsons",
}


class S3DataLoader:
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET", "documon-documents")
        self.endpoint_url = endpoint_url or os.getenv("AWS_ENDPOINT_URL_S3")
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region_name = region_name or os.getenv("AWS_REGION", "ap-southeast-1")

        if not self.bucket_name:
            raise ValueError("S3_BUCKET environment variable or parameter is missing.")

        print(f"[DEBUG] Initializing S3 Client for bucket '{self.bucket_name}' at endpoint '{self.endpoint_url}'")
        
        self.s3_client = boto3.client(
            "s3",
            region_name=self.region_name,
            endpoint_url=self.endpoint_url or None,
            aws_access_key_id=self.aws_access_key_id or None,
            aws_secret_access_key=self.aws_secret_access_key or None,
            config=Config(signature_version="s3v4"),
        )

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Generate a presigned GET URL for an S3 object key.
        """
        return self.s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": self.bucket_name,
                "Key": key,
            },
            ExpiresIn=expires_in,
        )

    def upload_fileobj(self, fileobj, key: str, content_type: Optional[str] = None) -> str:
        """
        Upload a file-like object to S3 at the given key and return its presigned download URL.
        """
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        print(f"[INFO] Uploading file to s3://{self.bucket_name}/{key}")
        self.s3_client.upload_fileobj(
            Fileobj=fileobj,
            Bucket=self.bucket_name,
            Key=key,
            ExtraArgs=extra_args if extra_args else None,
        )
        return self.generate_presigned_url(key)

    def upload_file(self, local_path: str, key: str, content_type: Optional[str] = None) -> str:
        """
        Upload a local file path to S3 at the given key and return its presigned download URL.
        """
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        print(f"[INFO] Uploading file '{local_path}' to s3://{self.bucket_name}/{key}")
        self.s3_client.upload_file(
            Filename=str(local_path),
            Bucket=self.bucket_name,
            Key=key,
            ExtraArgs=extra_args if extra_args else None,
        )
        return self.generate_presigned_url(key)


    def download_all_files(self, download_dir: str = "documents") -> List[FileMetadata]:
        """
        Download all objects from the S3 bucket into the specified base download directory,
        categorized into subfolders based on S3 key prefixes or file extensions.
        Returns a list of FileMetadata dicts.
        """
        base_path = Path(download_dir).resolve()
        base_path.mkdir(parents=True, exist_ok=True)
        downloaded_files: List[FileMetadata] = []

        print(f"[INFO] Listing objects in S3 bucket: '{self.bucket_name}'")
        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket_name)

        object_count = 0
        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key: str = obj["Key"]
                # Ignore folder placeholder objects
                if key.endswith("/"):
                    continue

                file_name = Path(key).name
                ext = Path(key).suffix.lower()
                doc_type = ext.lstrip(".") if ext else "unknown"

                # If key has a subfolder prefix (e.g. pdfs/file.pdf), use base_path / key.
                # Otherwise, determine subfolder from extension.
                if "/" in key:
                    local_file_path = base_path / key
                else:
                    subfolder = FOLDER_MAPPING.get(ext, doc_type + "s")
                    local_file_path = base_path / subfolder / file_name

                local_file_path.parent.mkdir(parents=True, exist_ok=True)

                download_url = self.generate_presigned_url(key)
                print(f"[INFO] Downloading s3://{self.bucket_name}/{key} -> {local_file_path}")
                self.s3_client.download_file(self.bucket_name, key, str(local_file_path))

                downloaded_files.append({
                    "local_path": local_file_path,
                    "name": file_name,
                    "type": doc_type,
                    "download_url": download_url,
                    "key": key,
                })
                object_count += 1

        print(f"[INFO] Downloaded {object_count} files from S3 bucket '{self.bucket_name}' into '{base_path}'.")
        return downloaded_files

    def load_all_documents(self, download_dir: str = "documents") -> List[Document]:
        """
        Downloads all documents from S3 into the download directory (categorized by file type),
        parses them with DataLoader, and returns the resulting LangChain Document list.
        """
        self.download_all_files(download_dir=download_dir)
        loader = DataLoader(download_dir)
        documents = loader.load_all_documents()
        return documents

