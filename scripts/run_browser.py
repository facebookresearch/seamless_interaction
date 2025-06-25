#!/usr/bin/env python3
"""
Launcher script for the Seamless Interaction Dataset Browser.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'boto3'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print(f"📦 Install them with: pip install -r requirements_browser.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_files():
    """Check if required files exist."""
    required_files = [
        'dataset_browser.py',
        'utils/fs.py',
        'requirements_browser.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files are present")
    return True

def check_metadata():
    """Check if metadata files exist."""
    metadata_paths = [
        'filelists/improvised/dev/metadata.csv',
        'filelists/improvised/train/metadata.csv',
        'filelists/improvised/test/metadata.csv'
    ]
    
    existing_files = []
    for file_path in metadata_paths:
        if Path(file_path).exists():
            existing_files.append(file_path)
    
    if existing_files:
        print(f"✅ Found {len(existing_files)} metadata files")
        return True
    else:
        print("⚠️  No metadata files found. The browser will still work for S3 browsing.")
        return True

def main():
    """Main launcher function."""
    print("🎭 Seamless Interaction Dataset Browser Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check files
    if not check_files():
        sys.exit(1)
    
    # Check metadata (non-critical)
    check_metadata()
    
    print("\n🚀 Starting the dataset browser...")
    print("📱 The browser will open in your default web browser")
    print("🌐 Typically available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'dataset_browser.py',
            '--server.address', 'localhost',
            '--server.port', '8501',
            '--browser.serverAddress', 'localhost'
        ])
    except KeyboardInterrupt:
        print("\n👋 Browser stopped by user")
    except Exception as e:
        print(f"❌ Error starting browser: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 