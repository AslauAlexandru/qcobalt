"""
Automated setup script for Python virtual environments with Marimo.

Handles virtual environment creation, package installation, and Marimo launch
"""

import subprocess
import sys
import platform
from pathlib import Path

# Configuration
CONFIG = ".config"
ENTRY_NOTEBOOK = "index.py"
NOTEBOOKS = "notebooks"
REQUIREMENTS = "requirements.txt"
VIRTUAL_ENV = ".virtualenv"


def get_executables(virtual_env: Path) -> tuple[Path, Path]:
	"""Get platform-specific paths for Python and pip executables."""
	if platform.system() == "Windows":
		python: Path = virtual_env / "Scripts" / "python.exe"
		pip: Path = virtual_env / "Scripts" / "pip.exe"
	else:
		python: Path = virtual_env / "bin" / "python"
		pip: Path = virtual_env / "bin" / "pip"
	return (python, pip)


def check_config_directory():
	"""Verify that the .config directory exists."""
	config: Path = Path(CONFIG)
	if not config.exists():
		raise RuntimeError(f"{CONFIG} directory not found.")
	if not config.is_dir():
		raise RuntimeError(f"{CONFIG} exists but is not a directory.")


def check_python_version():
	"""Verify Python version meets minimum requirements."""
	version = sys.version_info
	if version.major < 3 or (version.major == 3 and version.minor < 9):
		raise RuntimeError(
			f"Python 3.9+ required. Current version: {version.major}.{version.minor}"
		)
	print(f"Python {version.major}.{version.minor}.{version.micro} detected")


def create_virtual_environment(virtual_env: Path):
	"""Create virtual environment using venv module."""
	if virtual_env.exists():
		print("Virtual environment already exists")
		return

	print(f"Creating virtual environment in: {virtual_env}")
	try:
		subprocess.run(
			[sys.executable, "-m", "venv", str(virtual_env)],
			check=True,
			capture_output=True,
			text=True,
		)
		print("Virtual environment created successfully")
	except subprocess.CalledProcessError as e:
		raise RuntimeError(f"Failed to create virtual environment: {e.stderr}")


def install_packages(pip: Path, requirements: Path):
	"""Install packages from requirements file using virtual environment pip."""
	if not requirements.exists():
		print(f"No {requirements} file found. Skipping package installation.")
		return
	print(f"Installing packages from {requirements}")
	try:
		subprocess.run(
			[str(pip), "install", "-r", str(requirements)],
			check=True,
			capture_output=True,
			text=True,
		)
		print("Packages installed")
	except subprocess.CalledProcessError as e:
		raise RuntimeError(f"Failed to install packages: {e.stderr}")


def launch_notebook(python: Path, notebook: Path):
	"""Launch notebook using virtual environment Python."""
	if not notebook.exists():
		raise RuntimeError(f"Notebook not found: {notebook}")
	print(f"Launching notebook: {notebook}")
	try:
		subprocess.run(
			[str(python), "-m", "marimo", "edit", str(notebook)],
			check=True,
		)
	except subprocess.CalledProcessError as e:
		raise RuntimeError(f"Failed to launch notebook: {e.stderr}")
	except KeyboardInterrupt:
		print("\nNotebook session ended")


# Main execution
try:
	print("Starting QCobalt setup")
	print("=" * 50)

	# Combine paths
	virtual_env: Path = Path(CONFIG) / VIRTUAL_ENV
	requirements: Path = Path(CONFIG) / REQUIREMENTS
	entry_notebook: Path = Path(NOTEBOOKS) / ENTRY_NOTEBOOK

	# Get platform-specific executable paths
	(python, pip) = get_executables(virtual_env)

	# Execute setup steps
	check_config_directory()
	check_python_version()
	create_virtual_environment(virtual_env)
	install_packages(pip, requirements)

	print("=" * 50)
	print("Setup completed successfully!")
	print("=" * 50)

	# Launch notebook automatically
	launch_notebook(python, entry_notebook)

except RuntimeError as e:
	print(f"Setup failed: {e}")
	sys.exit(1)
except KeyboardInterrupt:
	print("\nSetup interrupted by user")
	sys.exit(1)
