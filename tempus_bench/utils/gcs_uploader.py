"""
GCS uploader utility for syncing benchmark results to Google Cloud Storage.

This module provides utilities to upload benchmark results (runs directory)
to GCS using Default Application Credentials (Workload Identity compatible).
"""

import os
import logging
from pathlib import Path
from typing import Optional

from google.cloud import storage
from google.auth import default


logger = logging.getLogger(__name__)


def upload_directory_to_gcs(
    local_dir: str,
    gcs_bucket: str,
    gcs_prefix: Optional[str] = None,
    skip_upload: bool = False,
) -> str:
    """
    Upload a local directory recursively to Google Cloud Storage.

    Args:
        local_dir (str): Local directory path to upload.
        gcs_bucket (str): GCS bucket name (without gs:// prefix).
        gcs_prefix (Optional[str]): Prefix/path in bucket where to upload.
            If None, uses the directory name.
        skip_upload (bool): If True, skip actual upload (for testing).

    Returns:
        str: GCS path where the directory was uploaded (gs://bucket/path).

    Raises:
        ValueError: If local_dir doesn't exist or is not a directory.
        RuntimeError: If upload fails.

    Example:
        >>> gcs_path = upload_directory_to_gcs(
        ...     "/app/runs/run_20240101-120000",
        ...     "tempus_bench_results",
        ...     "runs/run_20240101-120000"
        ... )
        >>> print(gcs_path)  # gs://tempus_bench_results/runs/run_20240101-120000
    """
    local_path = Path(local_dir)

    if not local_path.exists():
        raise ValueError(f"Local directory does not exist: {local_dir}")

    if not local_path.is_dir():
        raise ValueError(f"Path is not a directory: {local_dir}")

    # Determine GCS prefix
    if gcs_prefix is None:
        gcs_prefix = local_path.name

    # Ensure prefix doesn't start with /
    gcs_prefix = gcs_prefix.lstrip("/")

    gcs_path = f"gs://{gcs_bucket}/{gcs_prefix}"

    if skip_upload:
        logger.info(f"Skip upload enabled. Would upload {local_dir} to {gcs_path}")
        return gcs_path

    try:
        # Initialize GCS client with default credentials
        credentials, project = default()
        client = storage.Client(credentials=credentials, project=project)

        # Get bucket
        bucket = client.bucket(gcs_bucket)

        # Upload files recursively
        uploaded_count = 0
        for root, dirs, files in os.walk(local_dir):
            root_path = Path(root)
            relative_path = root_path.relative_to(local_path)

            for file in files:
                local_file_path = root_path / file

                # Construct GCS blob path
                if relative_path == Path("."):
                    blob_path = f"{gcs_prefix}/{file}"
                else:
                    blob_path = f"{gcs_prefix}/{relative_path}/{file}"

                # Normalize path separators for GCS
                blob_path = blob_path.replace("\\", "/")

                # Upload file
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(local_file_path))

                uploaded_count += 1
                if uploaded_count % 100 == 0:
                    logger.debug(f"Uploaded {uploaded_count} files...")

        logger.info(
            f"Successfully uploaded {uploaded_count} files from {local_dir} to {gcs_path}"
        )

        return gcs_path

    except Exception as e:
        raise RuntimeError(
            f"Failed to upload directory {local_dir} to {gcs_path}: {str(e)}"
        ) from e


def upload_run_results(
    run_path: str,
    gcs_bucket: str = "tempus_bench_results",
    skip_upload: bool = False,
) -> str:
    """
    Upload benchmark run results to GCS.

    Convenience function that uploads a run directory to the standard
    location in the GCS bucket.

    Args:
        run_path (str): Path to the run directory (e.g., runs/run_20240101-120000).
        gcs_bucket (str): GCS bucket name. Defaults to "tempus_bench_results".
        skip_upload (bool): If True, skip actual upload (for testing).

    Returns:
        str: GCS path where results were uploaded.

    Example:
        >>> gcs_path = upload_run_results("runs/run_20240101-120000")
        >>> print(gcs_path)  # gs://tempus_bench_results/runs/run_20240101-120000
    """
    run_path_obj = Path(run_path)

    # Extract run directory name (e.g., "run_20240101-120000")
    run_dir_name = run_path_obj.name

    # Construct GCS prefix
    if run_path_obj.parent.name == "runs":
        # If path is runs/run_xxx, upload to runs/run_xxx
        gcs_prefix = f"runs/{run_dir_name}"
    else:
        # Otherwise, just use the directory name
        gcs_prefix = run_dir_name

    return upload_directory_to_gcs(
        local_dir=str(run_path),
        gcs_bucket=gcs_bucket,
        gcs_prefix=gcs_prefix,
        skip_upload=skip_upload,
    )



