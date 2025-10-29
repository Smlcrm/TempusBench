import os
import subprocess

from pathlib import Path

from .paths import get_project_root

class CondaEnvManager:
    def __init__(self, name: str, python: str = None, requirements_path: str = None):
        """
        Manage a conda environment: verify or create, install dependencies.

        Args:
            name (str): Name of the environment.
            python (str): Python version (e.g., '3.11').
            requirements_path (str): Path to requirements.txt or .yml file.
        """
        self.env_name = name
        self.python_version = python
        self.requirements_path = requirements_path

        # Check if the conda environment already exists and has tempus_bench installed
        check_result = subprocess.run(
            f"conda run -n {self.env_name} python --version && conda run -n {self.env_name} python -c 'import tempus_bench'",
            shell=True, executable="/bin/bash",
            capture_output=True, text=True
        )

        if check_result.returncode == 0 and "Python" in check_result.stdout:
            for line in check_result.stdout.split('\n'):
                if "Python" in line:
                    self.python_version = line.strip().split("Python")[-1].strip()
                    break
            self._env_created = True
            self._installed = True
        else:
            # Environment doesn't exist or tempus_bench not installed
            self.create_env()
            self.install(self.requirements_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.delete()

    def create_env(self):
        # Create the conda environment - fail fast, no fallbacks
        result = subprocess.run([
            "conda", "create", "-y", "-n", self.env_name, f"python={self.python_version}"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create conda environment {self.env_name}: {result.stderr}")

        # Install tempus_bench package
        result = subprocess.run([
            "conda", "run", "-n", self.env_name, "pip", "install", "-e", str(get_project_root()),
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to install tempus_bench in conda environment {self.env_name}: {result.stderr}")

        self._env_created = True

    def install(self, requirements_path: str):
        if not requirements_path.endswith(".txt"):
            raise ValueError("Unknown requirements file type. Provide .txt")

        result = subprocess.run([
            "conda", "run", "-n", self.env_name, "pip", "install", "-r", requirements_path
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to install requirements {requirements_path} in conda environment {self.env_name}: {result.stderr}")

        self._installed = True

    def run(self, script: str, args: str = ""):
        """
        Run a Python script inside the conda environment.

        Args:
            script (str): Path to script to run, or module.
            args (str): Arguments string to pass to `python script`.
        """
        result = subprocess.run([
            "conda", "run", "-n", self.env_name, "python", script, args
        ], capture_output=True, text=True)

        if result.returncode != 0:
            error_msg = (
                f"Failed to run script ({script}) in conda env ({self.env_name}).\n"
                f"Exit code: {result.returncode}\n"
                f"Standard Output:\n{result.stdout}\n"
                f"Standard Error:\n{result.stderr}"
            )
            raise RuntimeError(error_msg)

        return result

    def delete(self):
        if self._env_created:
            subprocess.run([
                "conda", "remove", "-y", "-n", self.env_name, "--all"
            ], check=True)
            self._env_created = False; self._installed = False
