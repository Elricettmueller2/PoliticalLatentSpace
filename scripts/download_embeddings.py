#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download large embedding files from external storage.
This script handles downloading and verifying large embedding files.
"""

import os
import sys
import argparse
import requests
import hashlib
import time
from tqdm import tqdm


def calculate_md5(file_path, chunk_size=8192):
    """
    Calculate MD5 hash of a file.
    
    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read
        
    Returns:
        MD5 hash as hex string
    """
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def download_file(url, output_path, expected_md5=None):
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        expected_md5: Expected MD5 hash for verification
        
    Returns:
        Path to the downloaded file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if file already exists and has correct MD5
    if os.path.exists(output_path) and expected_md5:
        print(f"File already exists at {output_path}, checking MD5...")
        actual_md5 = calculate_md5(output_path)
        if actual_md5 == expected_md5:
            print(f"MD5 matches ({actual_md5}), skipping download")
            return output_path
        else:
            print(f"MD5 mismatch (expected {expected_md5}, got {actual_md5}), re-downloading")
    
    # Start download
    print(f"Downloading from {url} to {output_path}")
    start_time = time.time()
    
    # Stream download with progress bar
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        
        # Get file size if available
        file_size = int(response.headers.get('content-length', 0))
        
        # Set up progress bar
        progress_bar = tqdm(
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=os.path.basename(output_path)
        )
        
        # Download file in chunks
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
    
    end_time = time.time()
    download_time = end_time - start_time
    
    # Calculate download speed
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    speed_mbps = file_size_mb / download_time
    
    print(f"Download complete: {file_size_mb:.2f} MB in {download_time:.2f}s ({speed_mbps:.2f} MB/s)")
    
    # Verify MD5 if provided
    if expected_md5:
        print("Verifying MD5 hash...")
        actual_md5 = calculate_md5(output_path)
        if actual_md5 == expected_md5:
            print(f"MD5 verification successful: {actual_md5}")
        else:
            print(f"MD5 verification failed: expected {expected_md5}, got {actual_md5}")
            print("The downloaded file may be corrupted or incomplete.")
            sys.exit(1)
    
    return output_path


def download_embeddings(url, output_path, md5=None):
    """
    Download embeddings file from external storage.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        md5: Expected MD5 hash for verification
        
    Returns:
        Path to the downloaded file
    """
    return download_file(url, output_path, md5)


def main():
    parser = argparse.ArgumentParser(description='Download embeddings from external storage')
    parser.add_argument('--url', type=str, required=True, 
                        help='URL to download from')
    parser.add_argument('--output-path', type=str, required=True, 
                        help='Path to save the file')
    parser.add_argument('--md5', type=str, 
                        help='Expected MD5 hash for verification')
    args = parser.parse_args()
    
    download_embeddings(args.url, args.output_path, args.md5)


if __name__ == "__main__":
    main()
