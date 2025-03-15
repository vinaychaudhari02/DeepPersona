import os
import requests
import time
from urllib.parse import urlparse


def download_images(image_urls, search_query):
    """Downloads images into a folder named after the search query, with timestamps."""
    # Format search query to remove spaces & special characters
    safe_query = "".join(c for c in search_query if c.isalnum()).lower()
    save_path = os.path.join("images", safe_query)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    downloaded_paths = []

    for idx, url in enumerate(image_urls):
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            print(f"❌ Skipping invalid URL: {url}")
            continue  # Skip invalid URLs

        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()

            # Get current timestamp for unique file names
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            # Determine file format (default to .jpg if no format is found)
            ext = os.path.splitext(parsed_url.path)[-1].lower()
            if ext not in [".jpg", ".jpeg", ".png"]:
                ext = ".jpg"  # Default to .jpg if format is unknown

            file_path = os.path.join(save_path, f"{safe_query}_image{idx + 1}_{timestamp}{ext}")

            # **Ensure file is fully downloaded before proceeding**
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)

            downloaded_paths.append(file_path)
            print(f"✅ Downloaded: {file_path}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to download {url}: {e}")

    return downloaded_paths
