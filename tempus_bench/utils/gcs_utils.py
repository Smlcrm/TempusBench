"""
GCS utility functions for downloading files from Google Cloud Storage.

This module provides utilities to download files from GCS paths using
Default Application Credentials (Workload Identity compatible).
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from google.cloud import storage
from google.auth import default


def download_file_from_gcs(gcs_path: str, local_path: Optional[str] = None) -> str:
    """
    Download a file from Google Cloud Storage to local filesystem.

    Args:
        gcs_path (str): GCS path in format `gs://bucket/path/to/file.ext`
        local_path (Optional[str]): Local file path to save the file.
            If None, saves to a temporary file.

    Returns:
        str: Path to the downloaded local file.

    Raises:
        ValueError: If gcs_path is not in the correct format.
        RuntimeError: If download fails.

    Example:
        >>> local_file = download_file_from_gcs("gs://my-bucket/config.yaml")
        >>> local_file = download_file_from_gcs("gs://my-bucket/config.yaml", "/tmp/my-config.yaml")
    """
    if not gcs_path.startswith("gs://"):
        raise ValueError(
            f"Invalid GCS path format. Expected 'gs://bucket/path', got: {gcs_path}"
        )

    # Parse GCS path
    path_without_prefix = gcs_path[5:]  # Remove 'gs://'
    if "/" not in path_without_prefix:
        raise ValueError(
            f"Invalid GCS path format. Expected 'gs://bucket/path', got: {gcs_path}"
        )

    bucket_name = path_without_prefix.split("/", 1)[0]
    blob_name = path_without_prefix.split("/", 1)[1]

    # Determine local path
    if local_path is None:
        # Create temporary file
        file_extension = Path(gcs_path).suffix
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension, prefix="gcs_download_"
        )
        local_path = temp_file.name
        temp_file.close()

    # Ensure directory exists
    local_file_path = Path(local_path)
    local_file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize GCS client with default credentials
        credentials, project = default()
        client = storage.Client(credentials=credentials, project=project)

        # Get bucket and blob
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Download file
        blob.download_to_filename(str(local_file_path))

        return str(local_file_path)

    except Exception as e:
        raise RuntimeError(
            f"Failed to download file from {gcs_path} to {local_path}: {str(e)}"
        ) from e



