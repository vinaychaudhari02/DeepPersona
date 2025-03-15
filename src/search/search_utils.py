import time
from selenium.webdriver.common.by import By
from selenium import webdriver
from urllib.parse import quote, urlparse

def validate_url(url):
    """Ensures the URL has a valid scheme (http or https)."""
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        return "https://" + url  # Add https:// if missing
    return url

def get_search_result_urls_google(query, max_results=10):
    """Searches Google and returns a list of image URLs."""
    driver = webdriver.Chrome()
    try:
        url = f"https://www.google.com/search?q={quote(query)}&tbm=isch"
        print(f"🔍 Opening URL: {url}")
        driver.get(url)
        time.sleep(5)  # Allow page load

        urls = []
        search_results = driver.find_elements(By.CSS_SELECTOR, "img")

        for result in search_results[:max_results]:
            img_url = result.get_attribute("src")
            if img_url:
                urls.append(validate_url(img_url))  # Ensure URL is properly formatted

        print(f"✅ Found {len(urls)} Google image URLs")
        return urls

    except Exception as e:
        print(f"❌ Error: {e}")
        return []

    finally:
        driver.quit()


def get_search_result_urls_bing(query, max_results=10):
    """Searches Bing and returns a list of image URLs."""
    driver = webdriver.Chrome()
    try:
        url = f"https://www.bing.com/images/search?q={quote(query)}"
        print(f"🔍 Opening URL: {url}")
        driver.get(url)

        time.sleep(5)

        urls = []
        search_results = driver.find_elements(By.CSS_SELECTOR, "img.mimg")

        for result in search_results[:max_results]:
            img_url = result.get_attribute("src")
            if img_url:
                urls.append(validate_url(img_url))  # Ensure URL is properly formatted

        print(f"✅ Found {len(urls)} Bing image URLs")
        return urls

    except Exception as e:
        print(f"❌ Error: {e}")
        return []

    finally:
        driver.quit()
