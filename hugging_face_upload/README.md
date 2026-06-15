# TempusBench Hugging Face upload

Upload the TempusBench task catalog to the public dataset
[**Smlcrm/tempus_bench_tasks**](https://huggingface.co/datasets/Smlcrm/tempus_bench_tasks).

## Prerequisites

Use the `tempus-env` conda environment (or any env with the requirements below):

```bash
conda activate tempus-env
pip install -r hugging_face_upload/requirements.txt
huggingface-cli login
# or: set HF_TOKEN=<your-token>
```

Ensure `tempus_bench/tasks/` exists locally with all 30 tasks (10 univariate, 10
multivariate, 10 covariate), each with `task.yaml` and its CSV.

## Create dataset and upload

From the repository root:

```bash
python hugging_face_upload/upload_dataset.py --repo-id Smlcrm/tempus_bench_tasks
```

This will:

1. Validate that every `task.yaml` references an existing CSV
2. Create the public dataset under the **Smlcrm** organization (if it does not exist)
3. Stage files under `tasks/<category>/<task>/` plus a generated `README.md`
4. Upload to Hugging Face

### Dry run (no API calls)

```bash
python hugging_face_upload/upload_dataset.py --dry-run
```

### Replace remote snapshot

```bash
python hugging_face_upload/upload_dataset.py --purge-remote
```

## Auto-download in TempusBench

After upload, `run_benchmark` (and any code calling `get_tasks_dir()`) automatically
downloads missing task CSVs on first use with a terminal explanation and progress bar.

To disable auto-download (e.g. offline tests):

```bash
set TEMPUS_BENCH_SKIP_TASK_DOWNLOAD=1
```

## Remove tasks from git (one-time, before PR)

Task data is gitignored under `tempus_bench/tasks/`. Untrack without deleting local files:

```bash
git rm -r --cached tempus_bench/tasks
```

Commit the code changes; keep your local `tempus_bench/tasks/` folder for upload and development.
