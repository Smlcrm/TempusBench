"""
Conda environment management for isolated model execution.

This module provides the CondaEnvManager class for creating and managing conda
environments with specific Python versions and dependencies. This ensures each
model runs in isolation to avoid dependency conflicts.
"""

import os
import subprocess

from pathlib import Path

from .paths import get_project_root


class CondaEnvManager:
    """
    Manages conda environments for isolated model execution.

    The CondaEnvManager creates conda environments with specific Python versions,
    installs the tempus_bench package, and installs model-specific dependencies
    from requirements.txt files. It can verify existing environments or create
    new ones.

    Attributes:
        env_name (str): Name of the conda environment.
        python_version (str): Python version used in the environment.
        requirements_path (str): Path to requirements.txt file.
        _env_created (bool): Whether the environment was created in this session.
        _installed (bool): Whether dependencies were installed in this session.
    """

    def __init__(
        self, name: str, python: str, requirements_path: str, reinstall: bool = False
    ):
        """
        Initialize conda environment manager and verify or create environment.

        This method checks if the conda environment exists and is healthy. If it
        doesn't exist, is unhealthy, or reinstall is True, it creates a new
        environment and installs dependencies.

        Args:
            name (str): Name of the conda environment.
            python (str): Python version (e.g., '3.11' or '3.11.13').
            requirements_path (str): Path to requirements.txt file for model
                dependencies.
            reinstall (bool): If True, removes existing environment before creating
                new one. Defaults to False.

        Raises:
            RuntimeError: If environment creation or package installation fails.
        """
        self.env_name = name
        self.python_version = python
        self.requirements_path = requirements_path
        self._env_created = False
        self._installed = False

        # If reinstall requested, remove env if it exists
        if reinstall:
            subprocess.run(
                ["conda", "env", "remove", "-n", self.env_name, "-y"],
                capture_output=True,
                text=True,
            )

        # Check if the conda environment already exists and has tempus_bench installed
        check_result = subprocess.run(
            f"conda run -n {self.env_name} python --version && conda run  -n {self.env_name} python -c 'import tempus_bench'",
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
        )

        if (
            check_result.returncode == 0
            and "Python" in check_result.stdout
            and not reinstall
        ):
            for line in check_result.stdout.split("\n"):
                if "Python" in line:
                    self.python_version = line.strip().split("Python")[-1].strip()
                    break
            self._env_created = True
            self._installed = True
        else:
            # Environment doesn't exist, not healthy, or reinstall requested
            self.create_env()
            self.install(self.requirements_path)

    def __enter__(self):
        """
        Enter the context manager.

        Returns:
            CondaEnvManager: Returns self for use in with statements.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager, cleaning up the conda environment.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.

        Returns:
            None: Deletes the conda environment if it was created in this session.
        """
        self.delete()

    def create_env(self):
        """
        Create the conda environment and install tempus_bench package.

        This method creates a new conda environment with the specified Python
        version and installs the tempus_bench package in editable mode.

        Raises:
            RuntimeError: If environment creation or package installation fails.
        """
        # Create the conda environment - fail fast, no fallbacks
        result = subprocess.run(
            [
                "conda",
                "create",
                "-y",
                "-n",
                self.env_name,
                f"python={self.python_version}",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create conda environment {self.env_name}: {result.stderr}"
            )

        # Install tempus_bench package
        result = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                self.env_name,
                "pip",
                "install",
                "-e",
                str(get_project_root()),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to install tempus_bench in conda environment {self.env_name}: {result.stderr}"
            )

        self._env_created = True

    def install(self, requirements_path: str):
        """
        Install dependencies from requirements.txt file in the conda environment.

        Args:
            requirements_path (str): Path to requirements.txt file.

        Raises:
            ValueError: If requirements_path does not end with ".txt".
            RuntimeError: If dependency installation fails.
        """
        if not requirements_path.endswith(".txt"):
            raise ValueError("Unknown requirements file type. Provide .txt")

        result = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                self.env_name,
                "pip",
                "install",
                "-r",
                requirements_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to install requirements {requirements_path} in conda environment {self.env_name}: {result.stderr}"
            )

        self._installed = True

    def run(
        self, script: str | None = None, args: str = "", command: str | None = None
    ):
        """
        Run a Python script or command inside the conda environment.

        This method executes a Python script or arbitrary command within the
        conda environment and returns the result with stdout and stderr.

        Args:
            script (Optional[str]): Path to script to run, or module name (e.g.,
                "-m tempus_bench.pipeline.model_executor"). Mutually exclusive with command.
            args (str): Arguments string to pass to `python script`. Arguments are
                split by whitespace. Defaults to empty string.
            command (Optional[str]): Full command string to run. Mutually exclusive
                with script.

        Returns:
            subprocess.CompletedProcess: Result object with stdout, stderr, and
                returncode attributes.

        Raises:
            ValueError: If both script and command are provided, or neither is provided.
            RuntimeError: If command execution fails.
        """
        if script and command:
            raise ValueError("Cannot specify both 'script' and 'command' parameters")
        if not script and not command:
            raise ValueError("Must specify either 'script' or 'command' parameter")

        if command:
            # Run arbitrary command in conda environment
            result = subprocess.run(
                f"conda run -n {self.env_name} {command}",
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
            )
        else:
            # Run Python script with args
            # script is guaranteed to be not None here due to validation above
            if script is None:
                raise ValueError("script cannot be None when command is not provided")

            # Build command list: base command + script + split args
            cmd_list = ["conda", "run", "-n", self.env_name, "python", script]

            if args:
                # Split args string into separate list elements
                cmd_list.extend(args.split())

            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
            )

        if result.returncode != 0:
            target = command if command else f"{script} {args}"
            error_msg = (
                f"Failed to run {target} in conda env ({self.env_name}).\n"
                f"Exit code: {result.returncode}\n"
                f"Standard Output:\n{result.stdout}\n"
                f"Standard Error:\n{result.stderr}"
            )
            raise RuntimeError(error_msg)

        return result

    def delete(self):
        """
        Delete the conda environment if it was created in this session.

        This method removes the conda environment only if it was created during
        this session (not if it existed beforehand). This allows reuse of
        existing environments across runs.

        Raises:
            subprocess.CalledProcessError: If conda environment removal fails.
        """
        if self._env_created:
            subprocess.run(
                ["conda", "remove", "-y", "-n", self.env_name, "--all"], check=True
            )
            self._env_created = False
            self._installed = False
