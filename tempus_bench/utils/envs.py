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
            # Even if environment exists, verify requirements are installed
            # If verification fails, install requirements
            try:
                self._verify_all_requirements_installed(self.requirements_path)
                self._installed = True
            except RuntimeError:
                # Requirements are missing, install them
                self.install(self.requirements_path)
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

        # Special handling for toto model: install TensorFlow via conda-forge for macOS compatibility
        # Check if this is the toto requirements file
        is_toto = "toto" in requirements_path.lower()
        
        if is_toto:
            # Try installing TensorFlow via conda-forge first (better macOS compatibility)
            print(f"Installing TensorFlow via conda-forge for better macOS compatibility...")
            conda_result = subprocess.run(
                [
                    "conda",
                    "install",
                    "-n",
                    self.env_name,
                    "-c",
                    "conda-forge",
                    "tensorflow=2.16",
                    "tensorboard=2.16",
                    "-y",
                ],
                capture_output=True,
                text=True,
            )
            # Don't fail if conda install fails - fall back to pip
            if conda_result.returncode != 0:
                print(f"Warning: conda install of TensorFlow failed, will try pip: {conda_result.stderr}")

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

        # Verify all packages were actually installed
        self._verify_all_requirements_installed(requirements_path)

        # Verify that critical packages can actually be imported (catches dependency conflicts)
        self._verify_packages_importable(requirements_path)

        self._installed = True

    def _verify_all_requirements_installed(self, requirements_path: str):
        """
        Verify that all packages from requirements.txt were successfully installed.

        This method uses pip's built-in `pip list` command to check if all packages
        from requirements.txt are actually installed. This is more reliable than
        trying to import packages, as it directly queries pip's package database.

        This verification runs once immediately after installation to catch
        installation failures early.

        Args:
            requirements_path (str): Path to the requirements.txt file.

        Raises:
            RuntimeError: If any package from requirements.txt is not found in pip list.
        """
        if not os.path.exists(requirements_path):
            raise RuntimeError(
                f"Requirements file not found: {requirements_path}"
            )

        # Read requirements.txt and extract package names
        required_packages = []
        with open(requirements_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Skip comments, empty lines, and git packages
            if (
                not line
                or line.startswith('#')
                or line.startswith('git+')
                or "@git+" in line
            ):
                continue
            
            # Extract package name (before ==, >=, <=, >, <, etc.)
            # Handle various version specifiers
            package_name = (
                line.split('==')[0].split('>=')[0].split('<=')[0]
                .split('>')[0].split('<')[0].split('!=')[0]
                .split('~=')[0].strip()
            )
            
            # Strip extras notation (e.g., "gluonts[torch]" -> "gluonts", "huggingface_hub[cli]" -> "huggingface_hub")
            if '[' in package_name:
                package_name = package_name.split('[')[0].strip()
            
            if package_name:
                required_packages.append(package_name.lower())  # Normalize to lowercase

        if not required_packages:
            return

        # Get list of installed packages using pip list
        result = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                self.env_name,
                "pip",
                "list",
                "--format=json",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to get installed packages list from conda environment {self.env_name}: {result.stderr}"
            )

        # Parse JSON output to get installed package names
        try:
            import json
            installed_packages = json.loads(result.stdout)
            installed_package_names = {pkg['name'].lower() for pkg in installed_packages}
        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(
                f"Failed to parse pip list output from conda environment {self.env_name}: {e}"
            )

        # Special handling for packages that are part of other packages
        # tf-keras is part of tensorflow 2.16+, so if tensorflow is installed, tf-keras is available
        package_aliases = {
            'tf-keras': 'tensorflow',  # tf-keras is included in tensorflow 2.16+
        }
        
        # Check which required packages are missing
        missing_packages = []
        for pkg in required_packages:
            if pkg not in installed_package_names:
                # Check if this package is an alias for another installed package
                if pkg in package_aliases:
                    alias_pkg = package_aliases[pkg]
                    if alias_pkg in installed_package_names:
                        # The alias package is installed, so this package is available
                        continue
                # Package is truly missing
                missing_packages.append(pkg)
        
        if missing_packages:
            error_msg = (
                f"Verification failed: The following packages from {requirements_path} "
                f"were not found in the installed packages list for conda environment {self.env_name}:\n"
                + "\n".join(f"  - {pkg}" for pkg in missing_packages)
                + f"\n\nThis indicates the installation did not complete successfully. "
                f"Please check the installation logs above for errors."
            )
            raise RuntimeError(error_msg)

    def _verify_packages_importable(self, requirements_path: str):
        """
        Verify that critical packages from requirements.txt can actually be imported.
        
        This catches cases where packages are installed but not usable due to:
        - Dependency conflicts
        - Missing transitive dependencies
        - Architecture/version incompatibilities
        
        This is a supplement to _verify_all_requirements_installed which only checks
        if packages are in pip list, not if they're actually importable.
        
        Args:
            requirements_path (str): Path to the requirements.txt file.
            
        Raises:
            RuntimeError: If any critical package cannot be imported.
        """
        if not os.path.exists(requirements_path):
            return
        
        # Read requirements.txt and extract package names that should be importable
        # Skip packages that are not directly importable (e.g., tensorflow, tensorboard)
        importable_packages = []
        skip_packages = {'tensorflow', 'tensorboard', 'tensorflow-estimator', 
                        'tensorflow-io-gcs-filesystem', 'google-cloud-storage',
                        'google-auth', 'google-auth-oauthlib', 'requests-oauthlib'}
        
        with open(requirements_path, 'r') as f:
            for line in f:
                line = line.strip()
                if (
                    not line
                    or line.startswith('#')
                    or line.startswith('git+')
                    or "@git+" in line
                ):
                    continue
                
                # Extract package name
                package_name = (
                    line.split('==')[0].split('>=')[0].split('<=')[0]
                    .split('>')[0].split('<')[0].split('!=')[0]
                    .split('~=')[0].strip()
                )
                
                # Only test packages that are likely to be directly importable
                # and are not in the skip list
                if package_name and package_name.lower() not in skip_packages:
                    # Test if package name matches import name (usually the same)
                    import_name = package_name.lower().replace('-', '_')
                    importable_packages.append((package_name, import_name))
        
        # Test imports for critical packages (limit to avoid too many tests)
        # Focus on model-specific packages that are most likely to fail
        critical_packages = [pkg for pkg in importable_packages 
                           if any(keyword in pkg[0].lower() for keyword in 
                                  ['tabpfn', 'torch', 'transformers'])]
        
        # Also test core packages that might have conflicts
        core_packages = [pkg for pkg in importable_packages 
                        if pkg[0].lower() in ['numpy', 'pandas', 'scikit-learn', 'sklearn']]
        
        # Combine and limit to avoid too many tests
        packages_to_test = (critical_packages + core_packages)[:5]
        
        if not packages_to_test:
            return
        
        failed_imports = []
        for package_name, import_name in packages_to_test:
            # Handle special cases for import names
            if package_name.lower() == 'scikit-learn':
                import_cmd = "import sklearn"
            elif package_name.lower() == 'tabpfn':
                # Test the actual import that the model uses
                import_cmd = "from tabpfn import TabPFNRegressor"
            else:
                import_cmd = f"import {import_name}"
            
            # Try to import the package with a longer timeout for packages that load models
            timeout = 30 if 'tabpfn' in package_name.lower() else 10
            result = subprocess.run(
                [
                    "conda",
                    "run",
                    "-n",
                    self.env_name,
                    "python",
                    "-c",
                    import_cmd,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            if result.returncode != 0:
                # Check if it's a ModuleNotFoundError (the actual issue we're trying to catch)
                if "ModuleNotFoundError" in result.stderr or "No module named" in result.stderr:
                    failed_imports.append((package_name, result.stderr))
                # For other errors, log but don't fail (might be expected for some packages)
        
        if failed_imports:
            error_msg = (
                f"Import verification failed: The following packages from {requirements_path} "
                f"could not be imported in conda environment {self.env_name}:\n"
                + "\n".join(f"  - {pkg}: {err[:200]}" for pkg, err in failed_imports)
                + f"\n\nThis indicates the packages were installed but are not usable, "
                f"likely due to dependency conflicts or missing dependencies. "
                f"Please check the installation logs above for errors."
            )
            raise RuntimeError(error_msg)

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
