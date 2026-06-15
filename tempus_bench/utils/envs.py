"""
Conda environment management for isolated model execution.

This module provides the CondaEnvManager class for creating and managing conda
environments with specific Python versions and dependencies. This ensures each
model runs in isolation to avoid dependency conflicts.
"""

import os
import subprocess
import sys

from pathlib import Path

from .paths import get_project_root


def _shell_subprocess_kwargs() -> dict:
    """Use bash on Unix; default shell (cmd/PowerShell) on Windows."""
    if sys.platform == "win32":
        return {"shell": True}
    return {"shell": True, "executable": "/bin/bash"}


def _conda_import_check_cmd(env_name: str) -> str:
    """Shell one-liner verifying env python and tempus_bench import."""
    return (
        f"conda run -n {env_name} python --version && "
        f'conda run -n {env_name} python -c "import tempus_bench"'
    )


class CondaEnvManager:
    def _conda_cmd(self, *args: str) -> list[str]:
        """Build a cross-platform command that invokes conda."""
        if os.name == "nt":
            cmd_exe = os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe")
            return [cmd_exe, "/d", "/c", "conda", *args]
        return ["conda", *args]

    def _conda_env_health_check(self) -> subprocess.CompletedProcess:
        """Check whether the conda env can run Python and import tempus_bench."""
        result_version = subprocess.run(
            self._conda_cmd("run", "-n", self.env_name, "python", "--version"),
            capture_output=True,
            text=True,
        )
        if result_version.returncode != 0:
            return result_version
        result_import = subprocess.run(
            [
                *self._conda_cmd("run", "-n", self.env_name, "python"),
                "-c",
                "import tempus_bench",
            ],
            capture_output=True,
            text=True,
        )
        if result_import.returncode != 0:
            return result_import
        return result_version

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
        self._reinstall = reinstall
        self._env_created = False
        self._installed = False
        self._skip_conda = os.environ.get("RUN_WITHOUT_CONDA") == "1"
        if self._skip_conda:
            return

        # If reinstall requested, remove env if it exists
        if reinstall:
            subprocess.run(
                self._conda_cmd("env", "remove", "-n", self.env_name, "-y"),
                capture_output=True,
                text=True,
            )

        # Check if the conda environment already exists and has tempus_bench installed
        check_result = self._conda_env_health_check()

        # Check if environment exists and is healthy (i.e., 'conda run ... python --version'
        # and 'import tempus_bench' both succeed, and no reinstall requested)
        if (
            check_result.returncode == 0
            and "Python" in check_result.stdout
            and not reinstall
        ):
            # Parse the Python version from the output so that self.python_version reflects the real version in env
            for line in check_result.stdout.split("\n"):
                if "Python" in line:
                    self.python_version = line.strip().split("Python")[-1].strip()
                    break
            # Mark that the environment was not created in this session
            self._env_created = False  # Env existed; we did not create it
        else:
            # Environment doesn't exist, not healthy, or reinstall requested
            self.create_env()
            self.install(self.requirements_path)

    def _ensure_env_lazy(self) -> None:
        """
        When _skip_conda, we skipped env creation at init. Before running the model,
        ensure the conda env exists and has deps. Creates/installs if needed.
        """
        if not self._skip_conda:
            return
        check_result = self._conda_env_health_check()
        if (
            check_result.returncode == 0
            and "Python" in (check_result.stdout or "")
        ):
            return
        # Env doesn't exist or is unhealthy; create and install
        if self._reinstall:
            subprocess.run(
                self._conda_cmd("env", "remove", "-n", self.env_name, "-y"),
                capture_output=True,
                text=True,
            )
        self.create_env()
        self.install(self.requirements_path)
        self._env_created = False  # Don't delete on exit; env is for reuse

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
            self._conda_cmd(
                "create", "-y", "-n", self.env_name, f"python={self.python_version}"
            ),
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
                *self._conda_cmd("run", "-n", self.env_name, "pip"),
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
                *self._conda_cmd("run", "-n", self.env_name, "pip"),
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
        self,
        script: str | None = None,
        args: str = "",
        command: str | list[str] | None = None,
        verbose: bool = False,
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
            command (Optional[str | list[str]]): Full command to run. When a list is
                provided, elements are passed as exact argv tokens (recommended).
            verbose (bool): If True, stream stdout/stderr to the console in real time.
                Defaults to False.

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

        cwd = str(get_project_root())
        # When RUN_WITHOUT_CONDA=1: we skip conda env creation at init (avoids blocking),
        # but we still run the model via conda run so it gets model-specific deps (e.g.
        # chronos-forecasting). Using sys.executable would run with the parent's Python
        # which lacks model deps. We do a lazy env ensure here.
        if self._skip_conda and command:
            self._ensure_env_lazy()
            cmd_list = self._conda_cmd("run", "-n", self.env_name)
            if isinstance(command, str):
                cmd_list.extend(command.split())
            else:
                cmd_list.extend(command)
            if verbose:
                proc = subprocess.Popen(
                    cmd_list,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=cwd,
                    **_shell_subprocess_kwargs(),
                )
                stdout_lines = []
                for line in proc.stdout:
                    line = line.rstrip()
                    print(line, flush=True)
                    stdout_lines.append(line + "\n")
                proc.wait()
                result = subprocess.CompletedProcess(
                    args=cmd_list,
                    returncode=proc.returncode,
                    stdout="".join(stdout_lines),
                    stderr="",
                )
            else:
                result = subprocess.run(
                    cmd_list,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    **_shell_subprocess_kwargs(),
                )
            if verbose and result.stdout:
                print(result.stdout, end="", flush=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to run in conda env {self.env_name}: exit {result.returncode}\n"
                    f"stdout: {getattr(result, 'stdout', '')}\nstderr: {getattr(result, 'stderr', '')}"
                )
            return result

        if command:
            # Run arbitrary command in conda environment (cwd=project root; task pickles use absolute paths)
            cmd_list = self._conda_cmd("run", "-n", self.env_name)
            if isinstance(command, str):
                cmd_list.extend(command.split())
            else:
                cmd_list.extend(command)
            if verbose:
                proc = subprocess.Popen(
                    cmd_list,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=cwd,
                    **_shell_subprocess_kwargs(),
                )
                stdout_lines = []
                for line in proc.stdout:
                    line = line.rstrip()
                    print(line, flush=True)
                    stdout_lines.append(line + "\n")
                proc.wait()
                result = subprocess.CompletedProcess(
                    args=cmd_list,
                    returncode=proc.returncode,
                    stdout="".join(stdout_lines),
                    stderr="",
                )
            else:
                result = subprocess.run(
                    cmd_list,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    **_shell_subprocess_kwargs(),
                )
        else:
            # Run Python script with args
            # script is guaranteed to be not None here due to validation above
            if script is None:
                raise ValueError("script cannot be None when command is not provided")

            # Build command list: base command + script + split args
            cmd_list = [*self._conda_cmd("run", "-n", self.env_name, "python"), script]

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
        if self._skip_conda:
            return
        if not self._env_created:
            return
        subprocess.run(
            self._conda_cmd("remove", "-y", "-n", self.env_name, "--all"), check=True
        )
        self._env_created = False
        self._installed = False
