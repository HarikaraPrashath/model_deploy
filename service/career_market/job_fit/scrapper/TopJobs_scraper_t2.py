# -*- coding: utf-8 -*-
"""
Auto-generated from TopJobs_scraper_t2.ipynb
Converted from Jupyter Notebook to a Python script.
"""

# NOTE: This script contains the notebook's code cells in order.
# You may want to refactor into functions/modules for production use.


# === Cell 1 ===
# CELL 1: Install Chrome + Dependencies
import os
import sys
# !apt-get update -qq  # (notebook magic/command)
# !apt-get install -yqq wget unzip > /dev/null 2>&1  # (notebook magic/command)

# Install latest stable Google Chrome
# !wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -  # (notebook magic/command)
# !echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list  # (notebook magic/command)
# !apt-get update -qq  # (notebook magic/command)
# !apt-get install -yqq google-chrome-stable > /dev/null 2>&1  # (notebook magic/command)

# Get Chrome version and download matching driver
import re, subprocess

# Try to detect Chrome version on all major platforms (Windows paths included)
WINDOWS_CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
]

def detect_chrome_version():
    commands = [
        "google-chrome --version",    # Linux
        "chrome --version",           # macOS/Windows if on PATH
    ]
    for p in WINDOWS_CHROME_CANDIDATES:
        if p and os.path.exists(p):
            commands.append(f'"{p}" --version')

    for cmd in commands:
        out = subprocess.getoutput(cmd)
        m = re.search(r"\d+\.\d+\.\d+", out or "")
        if m:
            return m.group(), cmd
    return None, None

version, version_cmd = detect_chrome_version()
if version:
    print(f"Detected Chrome version: {version} (via {version_cmd})")
else:
    print("Chrome version not detected via CLI; continuing with webdriver_manager.")

# !wget -q https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/{version}/linux64/chromedriver-linux64.zip  # (notebook magic/command)
# !unzip -o chromedriver-linux64.zip  # (notebook magic/command)
# !chmod +x chromedriver-linux64/chromedriver  # (notebook magic/command)
# !mv chromedriver-linux64/chromedriver /usr/bin/chromedriver  # (notebook magic/command)

# Install selenium and other dependencies
# !pip install -q selenium beautifulsoup4 pandas pillow > /dev/null  # (notebook magic/command)
print("Chrome + Driver installed successfully!")


# === Cell 2 ===
import os, re, time, requests, zipfile
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import InvalidSessionIdException, SessionNotCreatedException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
import json

# --- CONFIG ---
# Keyword can be overridden via environment variable TOPJOBS_KEYWORD or CLI args.
# Example:
#   TOPJOBS_KEYWORD="data engineer" python TopJobs_scraper_t2.py
#   python TopJobs_scraper_t2.py "frontend developer"
KEYWORD = os.environ.get("TOPJOBS_KEYWORD", "software engineer")
if len(sys.argv) > 1:
    cli_kw = " ".join(sys.argv[1:]).strip()
    if cli_kw:
        KEYWORD = cli_kw
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCR_OUTPUT_ROOT = os.path.join(BASE_DIR, "scr_output")
OUTPUT_FOLDER = os.path.join(SCR_OUTPUT_ROOT, "topjobs_ads")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def clean_name(s):
    return re.sub(r'[^\w\- ]', '_', s.strip())[:100]

def find_chrome_binary():
    env_paths = [
        os.environ.get("CHROME_BINARY"),
        os.environ.get("GOOGLE_CHROME_SHIM"),
    ]
    candidates = env_paths + [p for p in WINDOWS_CHROME_CANDIDATES if p]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None

# --- Chrome options ---
options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')
options.add_argument('--disable-extensions')
options.add_argument('--disable-browser-side-navigation')

chrome_binary = find_chrome_binary()
if chrome_binary:
    options.binary_location = chrome_binary
    print(f"Using Chrome binary: {chrome_binary}")
else:
    print("Chrome binary not found in default locations; using system default lookup.")

CHROMEDRIVER_PATH = ChromeDriverManager().install()
MAX_JOBS_PER_BROWSER = int(os.environ.get("TOPJOBS_MAX_JOBS_PER_BROWSER", "75"))


def start_driver():
    try:
        service = Service(CHROMEDRIVER_PATH)
        new_driver = webdriver.Chrome(service=service, options=options)
        return new_driver, WebDriverWait(new_driver, 20)
    except (SessionNotCreatedException, WebDriverException) as e:
        print(f"Failed to start Chrome; ensure Chrome is installed and matches the driver. Details: {e}")
        raise


def stop_driver(active_driver):
    try:
        if active_driver:
            active_driver.quit()
    except Exception:
        pass


def is_invalid_session_error(exc):
    if isinstance(exc, InvalidSessionIdException):
        return True
    message = str(exc).lower()
    return "invalid session id" in message or "session deleted" in message


driver, wait = start_driver()

jobs_since_browser_start = 0

print(f"Searching for: {KEYWORD.upper()}\n")
driver.get("https://www.topjobs.lk/index.jsp")
time.sleep(2)

# Perform search
driver.find_element(By.ID, "txtKeyWord").clear()
driver.find_element(By.ID, "txtKeyWord").send_keys(KEYWORD)
driver.find_element(By.ID, "btnSearch").click()

try:
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table#table")))
    print("Results page loaded")
except:
    print("No results or site changed")
    driver.quit()
    raise

time.sleep(2)
soup = BeautifulSoup(driver.page_source, "html.parser")
rows = soup.select("table#table tbody tr[onclick*='createAlert']")
print(f"Found {len(rows)} job ads\n")

jobs = []
for i, row in enumerate(rows, 1):
    onclick = row.get("onclick", "")
    m = re.search(r"createAlert\('(\d+)','([^']+)','([^']+)','([^']+)','([^']+)'\)", onclick)
    if not m: continue
    rid, ac, jc, ec, _ = m.groups()

    pos = row.find("h2").get_text(strip=True) if row.find("h2") else "N/A"
    emp = row.find("h1").get_text(strip=True) if row.find("h1") else "N/A"
    ref = row.find_all("td")[1].get_text(strip=True)

    url = f"https://www.topjobs.lk/employer/JobAdvertismentServlet?rid={rid}&ac={ac}&jc={jc}&ec={ec}&pg=applicant/vacancybyfunctionalarea.jsp"

    jobs.append({"ref": ref, "pos": pos, "emp": emp, "url": url})
    print(f"{i:2d}. [{ref}] {pos} - {emp}")

# --- Download ads ---
metadata = []

for idx, job in enumerate(jobs, 1):
    print(f"\n{idx}/{len(jobs)} -> {job['pos']} ({job['ref']})")
    safe = f"{job['ref']}_{clean_name(job['pos'])}"

    job_data = {
        "ref": job["ref"],
        "position": job["pos"],
        "employer": job["emp"],
        "url": job["url"],
        "type": None,
        "files": []
    }

    if MAX_JOBS_PER_BROWSER > 0 and jobs_since_browser_start >= MAX_JOBS_PER_BROWSER:
        print("   Restarting browser to keep long scrape stable")
        stop_driver(driver)
        driver, wait = start_driver()
        jobs_since_browser_start = 0

    for attempt in range(2):
        try:
            driver.get(job["url"])
            time.sleep(2)

            # Wait for the remark div to load
            try:
                wait.until(EC.presence_of_element_located((By.ID, "remark")))
            except Exception as e:
                if is_invalid_session_error(e):
                    raise
                print(f"   ! Could not find job content div")
                break

            soup = BeautifulSoup(driver.page_source, "html.parser")
            remark_div = soup.find("div", {"id": "remark"})

            if not remark_div:
                print(f"   ! No remark div found")
                break

            # Check if job ad is posted as an image
            img_in_remark = remark_div.find("img")

            if img_in_remark and img_in_remark.get("src"):
                # IMAGE-BASED AD
                print(f"    Image-based ad detected")
                job_data["type"] = "image"

                src = img_in_remark.get("src")
                src = urljoin(job["url"], src)

                try:
                    r = requests.get(src, timeout=12)
                    if r.status_code == 200:
                        ext = src.split(".")[-1].split("?")[0].split("/")[-1][:5].lower()
                        ext = re.sub(r"[^a-z]", "", ext)
                        if ext not in {"png", "jpg", "jpeg", "gif", "webp", "bmp"}:
                            ext = "png"
                        path = f"{OUTPUT_FOLDER}/{safe}_ad.{ext}"
                        with open(path, "wb") as f:
                            f.write(r.content)
                        job_data["files"].append(os.path.basename(path))
                        print(f"   @ Image saved: {os.path.basename(path)}")
                    else:
                        print(f"   X Failed to download image (status {r.status_code})")
                except Exception as e:
                    print(f"   X Error downloading image: {e}")

            else:
                # TEXT-BASED AD
                print(f"    Text-based ad detected")
                job_data["type"] = "text"

                # Extract text content
                text_content = remark_div.get_text(separator="\n", strip=True)
                job_data["raw_text"] = text_content
                text_path = f"{OUTPUT_FOLDER}/{safe}_content.txt"
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
                job_data["files"].append(os.path.basename(text_path))
                print(f"   @ Text saved: {os.path.basename(text_path)}")

                # Take screenshot of the job ad section
                try:
                    element = driver.find_element(By.ID, "remark")

                    # Scroll element into view
                    driver.execute_script("arguments[0].scrollIntoView(true);", element)
                    time.sleep(1)

                    # Get element size and position
                    location = element.location
                    size = element.size

                    # Take full page screenshot
                    screenshot_path = f"{OUTPUT_FOLDER}/{safe}_screenshot.png"
                    driver.save_screenshot(screenshot_path)

                    # Crop to just the job ad element using Pillow
                    from PIL import Image
                    img = Image.open(screenshot_path)

                    # Crop with some padding
                    left = max(0, location['x'] - 10)
                    top = max(0, location['y'] - 10)
                    right = min(img.width, location['x'] + size['width'] + 10)
                    bottom = min(img.height, location['y'] + size['height'] + 10)

                    cropped = img.crop((left, top, right, bottom))
                    cropped.save(screenshot_path)

                    job_data["files"].append(os.path.basename(screenshot_path))
                    print(f"   @ Screenshot saved: {os.path.basename(screenshot_path)}")
                except Exception as e:
                    print(f"   X Error taking screenshot: {e}")

            metadata.append(job_data)
            jobs_since_browser_start += 1
            break

        except Exception as e:
            if is_invalid_session_error(e) and attempt == 0:
                print("   X Browser session died; restarting Chrome and retrying this job")
                stop_driver(driver)
                driver, wait = start_driver()
                jobs_since_browser_start = 0
                continue
            print(f"   X Error processing job: {e}")
            break

stop_driver(driver)

# Save metadata
metadata_path = f"{OUTPUT_FOLDER}/metadata.json"
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"\n Metadata saved: {metadata_path}")

print(f"SUMMARY:")
print(f"  Total jobs processed: {len(metadata)}")
print(f"  Image-based ads: {sum(1 for j in metadata if j['type'] == 'image')}")
print(f"  Text-based ads: {sum(1 for j in metadata if j['type'] == 'text')}")

print(f"\nOutput saved to: {OUTPUT_FOLDER}")
