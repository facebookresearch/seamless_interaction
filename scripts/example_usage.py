#!/usr/bin/env python3
"""
Example usage of the refactored construct_file_id_tar_archive_mapping.py 
script. This demonstrates different ways to run the script with various 
configurations.
"""

import subprocess
import sys
from pathlib import Path


def run_script_example():
    """Example of how to run the refactored script programmatically"""
    
    script_path = (
        Path(__file__).parent / "construct_file_id_tar_archive_mapping.py"
    )
    
    # Basic usage - process dev and test only (skip train for faster execution)
    print("Running basic example (dev/test only)...")
    cmd = [
        sys.executable, str(script_path),
        "--split-dirs", 'dev:/path/to/dev,test:/path/to/test',
        "--output", "example_output.csv",
        "--verbose"
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        print("✓ Script completed successfully!")
        print(f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Script failed with error: {e}")
        print(f"Error output: {e.stderr}")
    
    # Example with custom parameters
    print("\nRunning with custom parameters...")
    cmd_custom = [
        sys.executable, str(script_path),
        "--output", "custom_output.csv",
        "--processes", "4",
        "--split-dirs", 'dev:/path/to/custom/dev,test:/path/to/custom/test',
        "--verbose"
    ]
    
    print(f"Command would be: {' '.join(cmd_custom)}")
    print("(Not executing due to custom paths)")


def show_help():
    """Show the help for the script"""
    script_path = (
        Path(__file__).parent / "construct_file_id_tar_archive_mapping.py"
    )
    
    print("Available options for the script:")
    cmd = [sys.executable, str(script_path), "--help"]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error getting help: {e}")


if __name__ == "__main__":
    print("Construct File ID Tar Archive Mapping - Usage Examples")
    print("=" * 60)
    
    show_help()
    print("\n" + "=" * 60)
    
    # Uncomment the line below to run the actual example
    # run_script_example()
    
    print("\nTo run the script manually, use commands like:")
    print("python construct_file_id_tar_archive_mapping.py --verbose")
    print("python construct_file_id_tar_archive_mapping.py " +
          "--processes 8 --output my_output.csv") 