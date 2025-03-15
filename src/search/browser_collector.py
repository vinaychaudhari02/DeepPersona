import os
import json
from search.search_utils import get_search_result_urls_google, get_search_result_urls_bing
from search.image_downloader import download_images

def run_browser_collection(query):
    """Search Google and Bing for images, download images into query-based folders, and store URLs in a JSON file."""
    print(f"\n🔍 Running search for: {query}")

    image_urls = []

    print("🔵 Searching on Google...")
    google_urls = get_search_result_urls_google(query, max_results=50)  # ✅ Increased limit to 50
    image_urls.extend(google_urls)

    print("🟢 Searching on Bing...")
    bing_urls = get_search_result_urls_bing(query, max_results=50)  # ✅ Increased limit to 50
    image_urls.extend(bing_urls)

    image_urls = list(set(image_urls))  # Remove duplicates
    print(f"✅ Found {len(image_urls)} total images.")

    if not image_urls:
        print("⚠️ No images found for the given search.")
        return None

    # Ensure JSON directory exists
    json_folder = "data/json"
    os.makedirs(json_folder, exist_ok=True)

    json_path = os.path.join(json_folder, f"{query.replace(' ', '_')}.json")

    # Store only image URLs in the JSON file
    with open(json_path, "w") as f:
        json.dump({"query": query, "image_urls": image_urls}, f, indent=4)

    print(f"📁 JSON file saved: {json_path}")

    # Download images and ensure they are saved correctly
    if image_urls:
        print(f"📥 Downloading images to folder: images/{query.replace(' ', '_').lower()}")
        downloaded_paths = download_images(image_urls, query)

    return json_path  # ✅ Return only the JSON file path
