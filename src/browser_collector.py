import requests
import time
import json
import string
import os
import hashlib
from urllib.parse import quote
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# Import Selenium modules
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# ----------------- Web Scraping Setup -----------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0 Safari/537.36"
    )
}

# Global cache to avoid re-scraping duplicate URLs.
global_scrape_cache = {}

def download_image(url, folder):
    """Download an image from a URL and save it in the given folder."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            img_data = response.content
            filename = hashlib.md5(url.encode('utf-8')).hexdigest() + ".jpg"
            file_path = os.path.join(folder, filename)
            with open(file_path, "wb") as f:
                f.write(img_data)
            return file_path
    except Exception as e:
        print("Error downloading image from:", url, e)
    return None

def setup_selenium_driver():
    """Sets up a headless Chrome Selenium driver."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    return driver

def fetch_webpage_data(url, image_save_folder, driver):
    """
    Uses Selenium to load a webpage, extract its rendered text, and download images.
    Returns a dictionary with 'text' and 'images' (list of local file paths).
    """
    try:
        driver.get(url)
        time.sleep(3)  # Wait for dynamic content to load.
        rendered_html = driver.page_source
        soup = BeautifulSoup(rendered_html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        imgs = []
        for img in soup.find_all("img"):
            src = img.get("src")
            if src and src.startswith("http"):
                imgs.append(src)
        os.makedirs(image_save_folder, exist_ok=True)
        downloaded_images = []
        for img_url in imgs:
            local_path = download_image(img_url, image_save_folder)
            if local_path:
                downloaded_images.append(local_path)
        return {"text": text, "images": downloaded_images}
    except Exception as e:
        print(f"Error fetching webpage data from {url}: {e}")
        return {"text": "", "images": []}

def compute_average_similarity(texts):
    """Compute average pairwise similarity among texts."""
    similarities = []
    n = len(texts)
    for i in range(n):
        for j in range(i+1, n):
            sim = SequenceMatcher(None, texts[i], texts[j]).ratio()
            similarities.append(sim)
    return sum(similarities) / len(similarities) if similarities else 0

# --- Search Result URL Functions ---
# Use Selenium to load the search page for dynamic content.
def get_search_result_urls_google(query, max_results=20, driver=None):
    url = "https://www.google.com/search?q=" + quote(query)
    urls = []
    if driver:
        driver.get(url)
        time.sleep(3)  # Wait for JavaScript to render the results.
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for div in soup.find_all("div", class_="yuRUbf"):
            a = div.find("a")
            if a and a.get("href"):
                urls.append(a["href"])
                if len(urls) >= max_results:
                    break
    else:
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            for div in soup.find_all("div", class_="yuRUbf"):
                a = div.find("a")
                if a and a.get("href"):
                    urls.append(a["href"])
                    if len(urls) >= max_results:
                        break
        except Exception as e:
            print("Error fetching Google search results:", e)
    return urls

def get_search_result_urls_bing(query, max_results=20, driver=None):
    url = "https://www.bing.com/search?q=" + quote(query)
    urls = []
    if driver:
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for li in soup.find_all("li", class_="b_algo"):
            a = li.find("a")
            if a and a.get("href"):
                urls.append(a["href"])
                if len(urls) >= max_results:
                    break
    else:
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            for li in soup.find_all("li", class_="b_algo"):
                a = li.find("a")
                if a and a.get("href"):
                    urls.append(a["href"])
                    if len(urls) >= max_results:
                        break
        except Exception as e:
            print("Error fetching Bing search results:", e)
    return urls

def get_search_result_urls_yahoo(query, max_results=20):
    url = "https://search.yahoo.com/search?p=" + quote(query)
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
    except Exception as e:
        print("Error fetching Yahoo search results:", e)
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    urls = []
    for div in soup.find_all("div", class_="Sr"):
        a = div.find("a")
        if a and a.get("href"):
            urls.append(a["href"])
            if len(urls) >= max_results:
                break
    return urls

def process_search_engine(engine_name, query, get_urls_func, max_results=20, driver=None, global_cache=global_scrape_cache, base_img_folder=None):
    print(f"\nProcessing {engine_name} search...")
    urls = get_urls_func(query, max_results, driver) if driver else get_urls_func(query, max_results)
    print(f"Found {len(urls)} URLs on {engine_name}.")
    pages_data = []
    texts = []
    # Use the new base image folder if provided.
    image_save_folder = os.path.join(base_img_folder, engine_name) if base_img_folder else os.path.join("downloaded_images", engine_name)
    os.makedirs(image_save_folder, exist_ok=True)
    for url in urls:
        if url in global_cache:
            print(f"URL already processed; reusing data from: {url}")
            data = global_cache[url]
        else:
            print(f"Fetching data from: {url}")
            data = fetch_webpage_data(url, image_save_folder, driver)
            global_cache[url] = data
        if data["text"]:
            texts.append(data["text"])
        else:
            print(f"Unable to extract text from {url}.")
        pages_data.append({"url": url, "text": data["text"], "images": data["images"]})
        time.sleep(0.5)
    similarity_score = compute_average_similarity(texts)
    return {"pages": pages_data, "similarity_score": similarity_score}

# ----------------- SIFT & Homography Functions -----------------

def compute_homography(src_pts, dst_pts):
    """Compute homography using the normalized DLT algorithm."""
    src_norm, T_src = normalize_points(src_pts)
    dst_norm, T_dst = normalize_points(dst_pts)
    n_points = src_pts.shape[0]
    A = []
    for i in range(n_points):
        x, y = src_norm[i]
        xp, yp = dst_norm[i]
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_norm = h.reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    H /= H[2, 2]
    return H

def normalize_points(pts):
    """
    Normalize a set of 2D points so that the centroid is at the origin and the average
    distance from the origin is sqrt(2).
    """
    centroid = np.mean(pts, axis=0)
    shifted_pts = pts - centroid
    dists = np.sqrt(np.sum(shifted_pts**2, axis=1))
    mean_dist = np.mean(dists)
    scale = np.sqrt(2) / mean_dist if mean_dist != 0 else 1.0
    T = np.array([[scale,     0, -scale * centroid[0]],
                  [    0, scale, -scale * centroid[1]],
                  [    0,     0,                    1]])
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm_h = (T @ pts_h.T).T
    pts_norm = pts_norm_h[:, :2] / pts_norm_h[:, 2][:, np.newaxis]
    return pts_norm, T

def detect_and_match(img1, img2):
    """
    Detect features in two images using SIFT and match them using BFMatcher with Lowe's ratio test.
    Returns matched keypoint locations and good matches.
    """
    if img1 is None or img2 is None:
        print("One of the images is empty. Skipping matching.")
        return np.empty((0, 2)), np.empty((0, 2)), [], [], []
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return np.empty((0, 2)), np.empty((0, 2)), kp1, kp2, []
    
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    src_pts = np.array([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good_matches])
    return src_pts, dst_pts, kp1, kp2, good_matches

def compute_inlier_ratio(img1, img2, threshold=5.0):
    """
    Detect features and compute the robust homography via RANSAC.
    Returns the inlier ratio (inliers/total good matches).
    """
    if img1 is None or img2 is None:
        return 0.0
    src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match(img1, img2)
    if len(good_matches) < 4:
        return 0.0
    H, inlier_mask = ransac_homography(src_pts, dst_pts, num_iter=1000, threshold=threshold)
    if inlier_mask is None or len(good_matches) == 0:
        return 0.0
    inlier_ratio = np.sum(inlier_mask) / len(good_matches)
    return inlier_ratio

def ransac_homography(src_pts, dst_pts, num_iter=1000, threshold=5.0):
    """
    Compute a robust homography using RANSAC.
    Returns the best homography and a boolean inlier mask.
    """
    max_inliers = 0
    best_H = None
    best_inlier_mask = None
    n = src_pts.shape[0]
    if n < 4:
        return None, None
    for i in range(num_iter):
        indices = np.random.choice(n, 4, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]
        try:
            candidate_H = compute_homography(src_sample, dst_sample)
        except np.linalg.LinAlgError:
            continue
        src_pts_h = np.hstack([src_pts, np.ones((n, 1))])
        projected = candidate_H @ src_pts_h.T
        projected /= projected[2, :]
        projected_pts = projected[:2, :].T
        errors = np.linalg.norm(projected_pts - dst_pts, axis=1)
        inlier_mask = errors < threshold
        num_inliers = np.sum(inlier_mask)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inlier_mask = inlier_mask
            best_H = candidate_H
    if best_H is not None and np.sum(best_inlier_mask) >= 4:
        src_inliers = src_pts[best_inlier_mask]
        dst_inliers = dst_pts[best_inlier_mask]
        best_H = compute_homography(src_inliers, dst_inliers)
    return best_H, best_inlier_mask

def gather_downloaded_images(base_folder="~/data/images"):
    """Collect all downloaded image file paths from the /data/images folder recursively."""
    downloaded_images = []
    if not os.path.exists(base_folder):
        return downloaded_images
    for root, _, files in os.walk(base_folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                downloaded_images.append(os.path.join(root, filename))
    return downloaded_images

# ----------------- Main Script -----------------
if __name__ == "__main__":
    # Get name and city.
    name = input("Enter the person's name: ")
    city = input("Enter the city: ")
    query = f"{name} {city}"
    print(f"\nSearch Query: {query}\n")

    # Sanitize file names.
    def sanitize_filename(s):
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        return "".join(c for c in s if c in valid_chars).replace(" ", "_")
    name_clean = sanitize_filename(name)
    city_clean = sanitize_filename(city)
    folder_tag = f"{name_clean}_{city_clean}"

    import os

    # Get the repository root (one level up from the src folder)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Define the data folder path relative to the repo root.
    data_folder = os.path.join(BASE_DIR, "data")

    # Ensure the data folder exists.
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Ensure the "images" and "json" subfolders exist within the data folder.
    images_parent_folder = os.path.join(data_folder, "images")
    json_folder = os.path.join(data_folder, "json")
    os.makedirs(images_parent_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)

    # Create the base image folder for the current search using your folder_tag.
    base_image_folder = os.path.join(images_parent_folder, folder_tag)
    os.makedirs(base_image_folder, exist_ok=True)

    # Setup Selenium driver.
    driver = setup_selenium_driver()

    # Process search engines using Selenium for Google and Bing, and Requests for Yahoo.
    results = {}
    results["bing"] = process_search_engine("Bing", query, get_search_result_urls_bing, max_results=1000, driver=driver, base_img_folder=base_image_folder)
    results["google"] = process_search_engine("Google", query, get_search_result_urls_google, max_results=1000, driver=driver, base_img_folder=base_image_folder)
    results["yahoo"] = process_search_engine("Yahoo", query, get_search_result_urls_yahoo, max_results=1000, driver=None, base_img_folder=base_image_folder)

    final_data = {"query": query, "results": results}

    # Save JSON in folder "data/json" with filename based on name and city.
    json_filename = f"{folder_tag}.json"
    json_filepath = os.path.join(json_folder, json_filename)
    with open(json_filepath, "w") as f:
        json.dump(final_data, f, indent=4)
    print(f"\nFinal JSON saved as '{json_filepath}'.")
    driver.quit()

    # ----------------- SIFT-Based Feature Matching Section -----------------
    perform_feature_matching = input("\nDo you want to perform feature matching? (y/n): ").strip().lower()
    if perform_feature_matching == "y":
        input_image_path = input("Enter the path for the input image (or press Enter for pairwise matching): ").strip()
        downloaded_images = gather_downloaded_images(base_folder=base_image_folder)
        keep_images = set()
        threshold_ratio = 0.5  # Require at least 50% inlier ratio

        if input_image_path:
            if not os.path.exists(input_image_path):
                print("The input image does not exist. Exiting feature matching.")
            else:
                input_img = cv2.imread(input_image_path)
                print(f"\nMatching input image with {len(downloaded_images)} downloaded images...")
                for img_path in downloaded_images:
                    img = cv2.imread(img_path)
                    ratio = compute_inlier_ratio(input_img, img, threshold=7.0)
                    print(f"Inlier ratio between input image and {img_path}: {ratio:.2f}")
                    if ratio > threshold_ratio:
                        keep_images.add(img_path)
        else:
            print(f"\nPerforming pairwise matching among {len(downloaded_images)} downloaded images...")
            for i in range(len(downloaded_images)):
                for j in range(i+1, len(downloaded_images)):
                    img1 = cv2.imread(downloaded_images[i])
                    img2 = cv2.imread(downloaded_images[j])
                    ratio = compute_inlier_ratio(img1, img2, threshold=7.0)
                    if ratio > threshold_ratio:
                        keep_images.add(downloaded_images[i])
                        keep_images.add(downloaded_images[j])
                        print(f"Pair: {downloaded_images[i]} & {downloaded_images[j]} => Inlier Ratio: {ratio:.2f}")

        # ----------------- Delete Unnecessary Images -----------------
        print("\n=== Deleting Images with Low Feature Matching ===")
        for img_path in downloaded_images:
            if img_path not in keep_images:
                try:
                    os.remove(img_path)
                    print(f"Deleted: {img_path}")
                except Exception as e:
                    print(f"Error deleting {img_path}: {e}")

        print("\nFeature matching and cleanup complete.")
    else:
        print("\nSkipping feature matching as per user selection.")