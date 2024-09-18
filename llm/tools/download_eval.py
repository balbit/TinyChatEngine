import os
import argparse
import requests
import zipfile
import hashlib

def download_file(url, output_path):
    """Download file from a URL and save it to the specified path."""
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded file to {output_path}")

def extract_zip(zip_path, extract_to):
    """Extract a zip file to the specified directory."""
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def md5(file_path):
    """Calculate MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def confirm(prompt):
    """Prompt user for yes/no confirmation."""
    while True:
        reply = input(f"{prompt} [y/n]: ").lower().strip()
        if reply in ['y', 'yes']:
            return True
        elif reply in ['n', 'no']:
            return False

def main():
    parser = argparse.ArgumentParser(description="Download evaluation datasets.")
    parser.add_argument('--eval', required=True, choices=['vizwiz'], help='Specify which eval dataset to download.')
    parser.add_argument('--verify-md5', action='store_true', help='Verify the downloaded file using MD5 checksum.')
    args = parser.parse_args()

    eval_name = args.eval
    eval_base_path = os.path.join('evals', 'data', eval_name)

    if eval_name == 'vizwiz':
        download_url = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip"
        expected_md5 = 'caabbb339107521dda2def71ef625e06'
        zip_file_name = "vizwiz_test.zip"
        zip_path = os.path.join(eval_base_path, zip_file_name)
        
        # Check if the dataset already exists
        if os.path.exists(eval_base_path):
            print(f"The directory {eval_base_path} already exists.")
            if confirm("Do you want to overwrite it?"):
                print("Overwriting the existing dataset...")
            else:
                print("Exiting without downloading.")
                return

        os.makedirs(eval_base_path, exist_ok=True)
        
        download_file(download_url, zip_path)

        if args.verify_md5:
            print("Verifying MD5 checksum...")
            downloaded_md5 = md5(zip_path)
            if downloaded_md5 != expected_md5:
                print(f"MD5 checksum mismatch! Expected {expected_md5}, got {downloaded_md5}.")
                print("The file may be corrupted.")
                return
            else:
                print("MD5 checksum verified successfully.")

        extract_zip(zip_path, eval_base_path)

        # Remove the zip file after extraction
        os.remove(zip_path)
        print(f"Downloaded and extracted {eval_name} dataset successfully.")

if __name__ == '__main__':
    main()