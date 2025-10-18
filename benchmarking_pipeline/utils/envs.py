import os
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

        # Check if the conda environment already exists
        version_result = subprocess.run(
            f"conda run -n {self.env_name} python --version",
            shell=True, executable="/bin/bash",
            capture_output=True, text=True
        )

        if version_result.returncode == 0 and "Python" in version_result.stdout:
            self.python_version = version_result.stdout.strip().split()[-1]
            self._env_created = True; self._installed = True
        else:
            self.create_env()
            self.install(self.requirements_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.delete()

    def create_env(self):
        # Create the conda environment
        subprocess.run([
            "conda", "create", "-y", "-n", self.env_name, f"python={self.python_version}"
        ], check=True)
        subprocess.run([
            "conda", "run", "-n", self.env_name, "pip", "install", "-e", ROOT_DIR,
        ], check=True)
        self._env_created = True

    def install(self, requirements_path: str):
        if requirements_path.endswith(".txt"):
            install_cmd = (
                f"source $(conda info --base)/etc/profile.d/conda.sh && "
                f"conda activate {self.env_name} && "
                f"pip install -r {requirements_path}"
            )
        else:
            raise ValueError("Unknown requirements file type. Provide .txt")

        completed = subprocess.run(install_cmd, shell=True, executable="/bin/bash")
        if completed.returncode != 0:
            raise RuntimeError(f"Failed to install requirements {requirements_path} in conda environment {self.env_name}")

        self._installed = True

    def run(self, script: str, args: str = ""):
        """
        Run a Python script inside the conda environment.

        Args:
            script (str): Path to script to run, or module.
            args (str): Arguments string to pass to `python script`.
        """
        run_cmd = (
            f"source $(conda info --base)/etc/profile.d/conda.sh && "
            f"conda activate {self.env_name} && "
            f"python {script} {args}"
        )
        completed = subprocess.run(run_cmd, shell=True, executable="/bin/bash", text=True)
        if completed.returncode != 0:
            error_msg = (
                f"Failed to run script ({script}) in conda env ({self.env_name}).\n"
                f"Command: {run_cmd}\n"
                f"Exit code: {completed.returncode}\n"
                f"Standard Output:\n{completed.stdout}\n"
                f"Standard Error:\n{completed.stderr}"
            )
            raise RuntimeError(error_msg)

    def delete(self):
        if self._env_created:
            subprocess.run([
                "conda", "remove", "-y", "-n", self.env_name, "--all"
            ], check=True)
            self._env_created = False; self._installed = False
