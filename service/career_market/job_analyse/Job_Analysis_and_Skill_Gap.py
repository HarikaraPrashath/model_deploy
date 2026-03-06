#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import re
import time
import json
import requests
from io import BytesIO
from urllib.parse import urljoin
from collections import Counter
from typing import Dict, List, Set, Tuple


# In[2]:


def notebook_setup() -> None:
    """Optional setup for Colab/Linux notebooks."""
    try:
        get_ipython  # type: ignore[name-defined]
    except NameError:
        return

    print(" Installing dependencies...")
    get_ipython().system('apt-get update -qq')
    get_ipython().system('apt-get install -yqq wget unzip > /dev/null 2>&1')
    get_ipython().system('wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -')
    get_ipython().system('echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list')
    get_ipython().system('apt-get update -qq')
    get_ipython().system('apt-get install -yqq google-chrome-stable > /dev/null 2>&1')

    import subprocess
    chrome_version = subprocess.getoutput("google-chrome --version")
    version = re.search(r"\d+\.\d+\.\d+", chrome_version).group()
    print(f"Chrome version: {version}")

    get_ipython().system('wget -q https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/{version}/linux64/chromedriver-linux64.zip')
    get_ipython().system('unzip -o chromedriver-linux64.zip > /dev/null 2>&1')
    get_ipython().system('chmod +x chromedriver-linux64/chromedriver')
    get_ipython().system('mv chromedriver-linux64/chromedriver /usr/bin/chromedriver')

    get_ipython().system('apt-get install -yqq tesseract-ocr libtesseract-dev > /dev/null 2>&1')
    get_ipython().system('pip install -q selenium beautifulsoup4 pandas pillow pytesseract spacy nltk > /dev/null')
    get_ipython().system('python -m spacy download en_core_web_sm > /dev/null 2>&1')

    print("All dependencies installed!\n")


# In[3]:


# ------------Import Libraries
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import pytesseract
import spacy
import nltk

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError as exc:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' is not installed. "
        "Run: python -m spacy download en_core_web_sm"
    ) from exc

print("Libraries loaded successfully!\n")


# In[4]:


# -------------- Configuration & Sample Student Profile

# Configuration
KEYWORD = "software engineer"
OUTPUT_FOLDER = "topjobs_ads"

# Sample Student Profile (replace with actual profile from Component 1)
STUDENT_PROFILE = {
    "name": "John Doe",
    "education": {
        "degree": "BSc in Computer Science",
        "university": "University of Colombo",
        "year": "Final Year",
        "gpa": 3.5
    },
    "technical_skills": [
        "Python", "Java", "JavaScript", "HTML", "CSS",
        "MySQL", "Git", "Linux", "Object-Oriented Programming",
        "Data Structures", "Algorithms"
    ],
    "soft_skills": [
        "Communication", "Team Work", "Problem Solving", "Time Management"
    ],
    "certifications": [
        "Python Programming Certificate",
        "Web Development Bootcamp"
    ],
    "projects": [
        "E-commerce Website using Django",
        "Student Management System",
        "Portfolio Website"
    ],
    "experience": [
        {
            "role": "Intern Developer",
            "company": "Tech Startup",
            "duration": "3 months",
            "responsibilities": ["Bug fixing", "Feature development", "Testing"]
        }
    ],
    "interests": ["Web Development", "Machine Learning", "Mobile Apps"]
}

print("Student Profile Loaded:")
print(f"Name: {STUDENT_PROFILE['name']}")
print(f"Skills: {len(STUDENT_PROFILE['technical_skills'])} technical, {len(STUDENT_PROFILE['soft_skills'])} soft")
print(f"Experience: {len(STUDENT_PROFILE['experience'])} positions\n")



# In[5]:


# ----------------- Skill Extraction Utilities
class UniversalSkillExtractor:
    """
    Extracts skills from job descriptions across ANY domain
    without relying on predefined skill lists
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

        # Common job description noise words to filter out
        self.noise_words = {
            'experience', 'required', 'preferred', 'must', 'should', 'will',
            'work', 'working', 'ability', 'strong', 'excellent', 'good',
            'knowledge', 'understanding', 'skills', 'skill', 'proficient',
            'years', 'year', 'month', 'day', 'time', 'people', 'team',
            'position', 'role', 'job', 'company', 'candidate', 'etc',
            'including', 'related', 'various', 'similar', 'equivalent'
        }

        # Keywords that indicate skill sections in job ads
        self.skill_section_markers = [
            'requirements', 'qualifications', 'skills required',
            'technical skills', 'competencies', 'must have',
            'key skills', 'essential skills', 'desired skills',
            'job requirements', 'minimum qualifications'
        ]

        # Patterns that indicate skills
        self.skill_patterns = [
            r'proficiency in ([^.,;]+)',
            r'experience with ([^.,;]+)',
            r'knowledge of ([^.,;]+)',
            r'expertise in ([^.,;]+)',
            r'familiar with ([^.,;]+)',
            r'skilled in ([^.,;]+)',
            r'ability to use ([^.,;]+)',
            r'working knowledge of ([^.,;]+)',
        ]

    def extract_skill_sections(self, text: str) -> List[str]:
        """Extract text sections that likely contain skill requirements"""
        sections = []
        text_lower = text.lower()

        for marker in self.skill_section_markers:
            # Find text after each marker
            if marker in text_lower:
                start_idx = text_lower.index(marker)
                # Extract next 500 characters after marker
                section = text[start_idx:start_idx + 500]
                sections.append(section)

        # If no specific sections found, use entire text
        if not sections:
            sections = [text]

        return sections

    def extract_phrases_with_nlp(self, text: str) -> Set[str]:
        """Use NLP to extract meaningful noun phrases and entities"""
        doc = nlp(text[:100000])  # Limit for performance
        phrases = set()

        # Extract named entities (tools, technologies, organizations)
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'WORK_OF_ART', 'LAW']:
                clean = self.clean_phrase(ent.text)
                if clean and len(clean.split()) <= 4:  # Max 4 words
                    phrases.add(clean)

        # Extract noun chunks (potential skills/tools)
        for chunk in doc.noun_chunks:
            clean = self.clean_phrase(chunk.text)
            if clean and len(clean.split()) <= 4:
                phrases.add(clean)

        return phrases

    def extract_with_patterns(self, text: str) -> Set[str]:
        """Extract skills using regex patterns"""
        skills = set()
        text_lower = text.lower()

        for pattern in self.skill_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Split by common separators
                items = re.split(r'[,;&/]|\band\b|\bor\b', match)
                for item in items:
                    clean = self.clean_phrase(item)
                    if clean:
                        skills.add(clean)

        return skills

    def extract_capitalized_terms(self, text: str) -> Set[str]:
        """Extract capitalized terms (often tools, software, certifications)"""
        # Find words that are capitalized (potential acronyms or proper nouns)
        words = text.split()
        capitalized = set()

        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w\s-]', '', word)

            # Check if it's an acronym (2-6 uppercase letters)
            if re.match(r'^[A-Z]{2,6}$', clean_word):
                capitalized.add(clean_word)

            # Check if it's a capitalized term (not at sentence start)
            elif clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                capitalized.add(clean_word.lower())

        return capitalized

    def extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet point items (often skills)"""
        # Common bullet patterns
        bullet_patterns = [
            r'^\s*[-*]\s+(.+)',
            r'^\s*\d+\.\s+(.+)',
            r'^\s*[•·]\s+(.+)'
        ]

        bullets = []
        for line in text.split('\n'):
            for pattern in bullet_patterns:
                match = re.match(pattern, line)
                if match:
                    bullets.append(match.group(1).strip())
                    break

        return bullets

    def clean_phrase(self, phrase: str) -> str:
        """Clean and normalize extracted phrases"""
        # Convert to lowercase
        phrase = phrase.lower().strip()

        # Remove leading/trailing punctuation
        phrase = re.sub(r'^[^\w]+|[^\w]+$', '', phrase)

        # Remove possessives
        phrase = re.sub(r"'s\b", '', phrase)

        # Skip if too short or too long
        if len(phrase) < 2 or len(phrase) > 50:
            return ''

        # Skip if it's a noise word
        if phrase in self.noise_words or phrase in self.stop_words:
            return ''

        # Skip if it's all numbers
        if phrase.replace(' ', '').isdigit():
            return ''

        return phrase

    def extract_skills(self, text: str) -> Set[str]:
        """
        Main method: Extract all potential skills from text
        Combines multiple extraction techniques
        """
        all_skills = set()

        # 1. Extract from skill sections
        sections = self.extract_skill_sections(text)
        for section in sections:
            # Use NLP on sections
            all_skills.update(self.extract_phrases_with_nlp(section))
            # Use patterns
            all_skills.update(self.extract_with_patterns(section))

        # 2. Extract capitalized terms (acronyms, software names)
        all_skills.update(self.extract_capitalized_terms(text))

        # 3. Extract from bullet points
        bullets = self.extract_bullet_points(text)
        for bullet in bullets:
            clean = self.clean_phrase(bullet)
            if clean and len(clean.split()) <= 5:
                all_skills.add(clean)

        # 4. Use patterns on full text
        all_skills.update(self.extract_with_patterns(text))

        # Filter out remaining noise
        filtered_skills = {
            skill for skill in all_skills
            if skill and not all(word in self.stop_words for word in skill.split())
        }

        return filtered_skills

    def categorize_skills_automatically(self, all_job_skills: List[Set[str]]) -> Dict[str, List[str]]:
        """
        Automatically categorize skills based on frequency and co-occurrence
        across multiple job descriptions
        """
        # Count skill frequency
        skill_counter = Counter()
        for job_skills in all_job_skills:
            skill_counter.update(job_skills)

        # Get most common skills (these are likely important for the field)
        common_skills = [skill for skill, count in skill_counter.most_common(50)]

        # Simple categorization based on keywords
        categories = {
            'technical_tools': [],
            'software_skills': [],
            'certifications': [],
            'methodologies': [],
            'soft_skills': [],
            'domain_knowledge': [],
            'other': []
        }

        # Keywords for categorization
        tool_keywords = ['software', 'tool', 'platform', 'system', 'application']
        cert_keywords = ['certification', 'certified', 'license', 'accreditation']
        method_keywords = ['methodology', 'approach', 'framework', 'method']
        soft_keywords = ['communication', 'leadership', 'teamwork', 'management',
                         'problem solving', 'analytical', 'organizational']

        for skill in common_skills:
            skill_lower = skill.lower()

            if any(kw in skill_lower for kw in soft_keywords):
                categories['soft_skills'].append(skill)
            elif any(kw in skill_lower for kw in cert_keywords):
                categories['certifications'].append(skill)
            elif any(kw in skill_lower for kw in method_keywords):
                categories['methodologies'].append(skill)
            elif any(kw in skill_lower for kw in tool_keywords):
                categories['technical_tools'].append(skill)
            elif skill.isupper() or skill[0].isupper():
                categories['software_skills'].append(skill)
            else:
                categories['domain_knowledge'].append(skill)

        # Remove empty categories
        return {k: v for k, v in categories.items() if v}


#  Helper Functions

def extract_experience_years(text: str) -> int:
    """
    Extract required years of experience from job description

    Args:
        text: Job description text

    Returns:
        Number of years required (0 if not specified)
    """
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?).*?(?:experience|exp)',
        r'(?:experience|exp).*?(\d+)\+?\s*(?:years?|yrs?)',
        r'minimum.*?(\d+)\+?\s*(?:years?|yrs?)',
        r'at least.*?(\d+)\+?\s*(?:years?|yrs?)',
        r'(\d+)\s*(?:to|-)\s*\d+\s*(?:years?|yrs?)',
    ]

    text_lower = text.lower()

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue

    return 0  # No specific experience mentioned


#  Usage Example

def analyze_jobs_universal(job_texts: List[str], student_profile: Dict) -> Dict:
    """
    Analyze jobs for ANY field and compare with student profile

    Args:
        job_texts: List of job description texts
        student_profile: Student's profile with skills

    Returns:
        Complete analysis with skill gaps and matches
    """

    extractor = UniversalSkillExtractor()

    # Extract skills from all job descriptions
    print("Extracting skills from job descriptions...")
    all_job_skills = []
    job_analyses = []

    for i, text in enumerate(job_texts, 1):
        skills = extractor.extract_skills(text)
        all_job_skills.append(skills)

        job_analyses.append({
            'job_index': i,
            'required_skills': list(skills),
            'total_skills': len(skills)
        })

        print(f"  Job {i}: Found {len(skills)} skills")

    # Aggregate all unique skills across jobs
    all_unique_skills = set()
    skill_frequency = Counter()

    for skills in all_job_skills:
        all_unique_skills.update(skills)
        skill_frequency.update(skills)

    print(f"\nTotal unique skills found: {len(all_unique_skills)}")

    # Categorize skills automatically
    categories = extractor.categorize_skills_automatically(all_job_skills)

    # Normalize student skills
    student_skills = set()
    for skill_list in student_profile.values():
        if isinstance(skill_list, list):
            student_skills.update([s.lower().strip() for s in skill_list])
        elif isinstance(skill_list, str):
            student_skills.add(skill_list.lower().strip())

    print(f"Student has {len(student_skills)} skills in profile\n")

    # Compare with each job
    job_matches = []
    for i, job_skills in enumerate(all_job_skills, 1):
        matched = student_skills.intersection(job_skills)
        missing = job_skills - student_skills

        match_pct = (len(matched) / len(job_skills) * 100) if job_skills else 0

        job_matches.append({
            'job_index': i,
            'match_percentage': round(match_pct, 2),
            'matched_skills': list(matched),
            'missing_skills': list(missing),
            'total_required': len(job_skills)
        })

    # Find most common missing skills
    all_missing = Counter()
    for match in job_matches:
        all_missing.update(match['missing_skills'])

    # Analysis summary
    analysis = {
        'total_jobs': len(job_texts),
        'total_unique_skills': len(all_unique_skills),
        'skill_categories': categories,
        'most_required_skills': [
            {'skill': skill, 'frequency': count}
            for skill, count in skill_frequency.most_common(20)
        ],
        'student_skills_count': len(student_skills),
        'job_matches': sorted(job_matches, key=lambda x: x['match_percentage'], reverse=True),
        'top_missing_skills': [
            {'skill': skill, 'frequency': count}
            for skill, count in all_missing.most_common(15)
        ],
        'average_match': round(sum(j['match_percentage'] for j in job_matches) / len(job_matches), 2)
    }

    return analysis



# In[6]:


# -------------OCR Processing

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """Preprocess image to improve OCR accuracy"""
    # Convert to grayscale
    img = img.convert('L')

    # Increase contrast
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)

    # Resize if too small
    if img.width < 1000:
        scale = 1000 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)))

    return img


def perform_ocr(img: Image.Image) -> str:
    """Perform OCR on image and return extracted text"""
    try:
        processed = preprocess_image_for_ocr(img)
        # Configure Tesseract
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed, config=custom_config)
        return text
    except Exception as e:
        print(f"    OCR Error: {e}")
        return ""


def _storage_enabled() -> bool:
    return bool(
        os.environ.get("SUPABASE_URL", "").strip()
        and os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        and os.environ.get("SUPABASE_STORAGE_BUCKET", "").strip()
    )


def _storage_object_url(remote_path: str) -> str:
    url = os.environ.get("SUPABASE_URL", "").strip()
    bucket = os.environ.get("SUPABASE_STORAGE_BUCKET", "").strip()
    return f"{url}/storage/v1/object/{bucket}/{remote_path}"


def _storage_public_url(remote_path: str) -> str:
    url = os.environ.get("SUPABASE_URL", "").strip()
    bucket = os.environ.get("SUPABASE_STORAGE_BUCKET", "").strip()
    return f"{url}/storage/v1/object/public/{bucket}/{remote_path}"


def _upload_bytes_to_storage(content: bytes, remote_path: str, content_type: str) -> str | None:
    if not _storage_enabled():
        return None
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    headers = {
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    try:
        resp = requests.post(_storage_object_url(remote_path), data=content, headers=headers, timeout=30)
        if resp.status_code in (200, 201):
            return _storage_public_url(remote_path)
        return None
    except Exception:
        return None


def _upload_json_to_storage(data: object, remote_path: str) -> str | None:
    try:
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    except Exception:
        return None
    return _upload_bytes_to_storage(payload, remote_path, "application/json")


def _upload_text_to_storage(text: str, remote_path: str) -> str | None:
    return _upload_bytes_to_storage(text.encode("utf-8"), remote_path, "text/plain")


def _analysis_path(prefix: str | None, name: str) -> str | None:
    if not prefix:
        return None
    return f"{prefix.rstrip('/')}/{name}"

print(" OCR utilities loaded\n")


# In[7]:


# -------------- Web Scraping Functions

def clean_name(s: str) -> str:
    """Clean filename"""
    return re.sub(r'[^\w\- ]', '_', s.strip())[:100]

def scrape_topjobs(
    keyword: str,
    output_folder: str,
    write_local: bool = True,
    storage_prefix: str | None = None,
) -> List[Dict]:
    """Scrape job ads from TopJobs.lk"""

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')

    if write_local:
        os.makedirs(output_folder, exist_ok=True)

    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 20)

    print(f" Searching for: {keyword.upper()}")
    driver.get("https://www.topjobs.lk/index.jsp")
    time.sleep(3)

    # Perform search
    driver.find_element(By.ID, "txtKeyWord").clear()
    driver.find_element(By.ID, "txtKeyWord").send_keys(keyword)
    driver.find_element(By.ID, "btnSearch").click()

    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table#table")))
    except:
        print(" No results found")
        driver.quit()
        return []

    time.sleep(4)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    rows = soup.select("table#table tbody tr[onclick*='createAlert']")
    print(f" Found {len(rows)} job ads\n")

    jobs = []
    for i, row in enumerate(rows, 1):
        onclick = row.get("onclick", "")
        m = re.search(r"createAlert\('(\d+)','([^']+)','([^']+)','([^']+)','([^']+)'\)", onclick)
        if not m:
            continue

        rid, ac, jc, ec, _ = m.groups()
        pos = row.find("h2").get_text(strip=True) if row.find("h2") else "N/A"
        emp = row.find("h1").get_text(strip=True) if row.find("h1") else "N/A"
        ref = row.find_all("td")[1].get_text(strip=True)
        url = f"https://www.topjobs.lk/employer/JobAdvertismentServlet?rid={rid}&ac={ac}&jc={jc}&ec={ec}&pg=applicant/vacancybyfunctionalarea.jsp"

        jobs.append({"ref": ref, "pos": pos, "emp": emp, "url": url})
        print(f"  {i:2d}. [{ref}] {pos}  {emp}")

    # Download job details
    metadata = []
    for idx, job in enumerate(jobs, 1):
        print(f"\n {idx}/{len(jobs)}  {job['pos']} ({job['ref']})")
        safe = f"{job['ref']}_{clean_name(job['pos'])}"

        job_data = {
            "ref": job["ref"],
            "position": job["pos"],
            "employer": job["emp"],
            "url": job["url"],
            "type": None,
            "files": [],
            "raw_text": ""
        }

        try:
            driver.get(job["url"])
            time.sleep(4)

            wait.until(EC.presence_of_element_located((By.ID, "remark")))
            soup = BeautifulSoup(driver.page_source, "html.parser")
            remark_div = soup.find("div", {"id": "remark"})

            if not remark_div:
                continue

            img_in_remark = remark_div.find("img")

            if img_in_remark and img_in_remark.get("src"):
                # Image-based ad
                print("    Image-based ad")
                job_data["type"] = "image"

                src = urljoin(job["url"], img_in_remark.get("src"))
                try:
                    r = requests.get(src, timeout=12)
                    if r.status_code == 200:
                        ext = src.split(".")[-1].split("?")[0][:4] or "png"
                        content = r.content
                        if write_local:
                            path = f"{output_folder}/{safe}_ad.{ext}"
                            open(path, "wb").write(content)
                            job_data["files"].append(os.path.basename(path))
                            print(f"    Image saved: {os.path.basename(path)}")
                        if not write_local:
                            try:
                                img = Image.open(BytesIO(content))
                                job_data["raw_text"] = perform_ocr(img)
                            except Exception:
                                job_data["raw_text"] = job_data.get("raw_text", "")
                        remote_path = _analysis_path(storage_prefix, f"{safe}_ad.{ext}")
                        if remote_path:
                            image_url = _upload_bytes_to_storage(content, remote_path, f"image/{ext}")
                            if image_url:
                                job_data["image_url"] = image_url
                except Exception as e:
                    print(f"    Image download failed: {e}")
            else:
                # Text-based ad
                print("    Text-based ad")
                job_data["type"] = "text"

                text_content = remark_div.get_text(separator="\n", strip=True)
                job_data["raw_text"] = text_content
                if write_local:
                    text_path = f"{output_folder}/{safe}_content.txt"
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(text_content)
                    job_data["files"].append(os.path.basename(text_path))
                    print(f"    Text saved")

                    # Screenshot
                    try:
                        element = driver.find_element(By.ID, "remark")
                        driver.execute_script("arguments[0].scrollIntoView(true);", element)
                        time.sleep(1)

                        screenshot_path = f"{output_folder}/{safe}_screenshot.png"
                        driver.save_screenshot(screenshot_path)

                        location = element.location
                        size = element.size
                        img = Image.open(screenshot_path)

                        left = max(0, location['x'] - 10)
                        top = max(0, location['y'] - 10)
                        right = min(img.width, location['x'] + size['width'] + 10)
                        bottom = min(img.height, location['y'] + size['height'] + 10)

                        cropped = img.crop((left, top, right, bottom))
                        cropped.save(screenshot_path)
                        job_data["files"].append(os.path.basename(screenshot_path))
                        print(f"    Screenshot saved")
                    except Exception as e:
                        print(f"    Screenshot failed: {e}")

            metadata.append(job_data)

        except Exception as e:
            print(f"    Error: {e}")
            continue

    driver.quit()
    return metadata

print("Scraping functions loaded\n")


# In[8]:


# ------------------ Job Requirement Analysis

def analyze_job_requirements(
    metadata: List[Dict],
    output_folder: str,
    write_local: bool = True,
    storage_prefix: str | None = None,
) -> List[Dict]:
    """Process all jobs: OCR images, extract skills and requirements"""

    print("ANALYZING JOB REQUIREMENTS")

    # Initialize universal extractor
    extractor = UniversalSkillExtractor()

    analyzed_jobs = []

    for idx, job in enumerate(metadata, 1):
        print(f"Analyzing {idx}/{len(metadata)}: {job['position']}")

        job_text = job.get("raw_text", "")

        # If scraper stored text in files (e.g., metadata from TopJobs_scraper_t2),
        # load the first available text file when raw_text is missing.
        if write_local and not job_text:
            for file in job.get("files", []):
                if str(file).lower().endswith(".txt"):
                    text_path = os.path.join(output_folder, file)
                    if os.path.exists(text_path):
                        try:
                            with open(text_path, "r", encoding="utf-8") as f:
                                job_text = f.read()
                        except Exception:
                            job_text = ""
                        if job_text:
                            break

        # Perform OCR on images when text is missing
        if job.get("type") == "image" and not job_text:
            if write_local:
                for file in job.get("files", []):
                    if file.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                        print(f"   Performing OCR on {file}...")
                        image_path = os.path.join(output_folder, file)
                        try:
                            img = Image.open(image_path)
                            ocr_text = perform_ocr(img)
                        except Exception:
                            ocr_text = ""
                        job_text += "\n" + ocr_text

                        ocr_path = image_path.replace(os.path.splitext(image_path)[1], '_ocr.txt')
                        with open(ocr_path, 'w', encoding='utf-8') as f:
                            f.write(ocr_text)
                        print(f"    OCR text saved: {os.path.basename(ocr_path)}")
            else:
                image_url = job.get("image_url")
                if isinstance(image_url, str) and image_url.startswith("http"):
                    try:
                        resp = requests.get(image_url, timeout=20)
                        if resp.status_code == 200:
                            img = Image.open(BytesIO(resp.content))
                            job_text += "\n" + perform_ocr(img)
                    except Exception:
                        pass

        # Extract skills using universal extractor
        required_skills = extractor.extract_skills(job_text)
        experience_years = extract_experience_years(job_text)

        analysis = {
            "ref": job["ref"],
            "position": job["position"],
            "employer": job["employer"],
            "url": job["url"],
            "type": job["type"],
            "required_skills": list(required_skills),
            "total_skills": len(required_skills),
            "experience_years": experience_years,
            "full_text": job_text[:5000]
        }

        analyzed_jobs.append(analysis)

        print(f"    Found {len(required_skills)} skills")
        print(f"    Experience required: {experience_years} years\n")

    # Auto-categorize skills after analyzing all jobs
    all_skills_sets = [set(job['required_skills']) for job in analyzed_jobs]
    skill_categories = extractor.categorize_skills_automatically(all_skills_sets)

    # Add categorization to each job
    for job in analyzed_jobs:
        job['skills_by_category'] = {}
        for category, skills_list in skill_categories.items():
            matching = [s for s in job['required_skills'] if s in skills_list]
            if matching:
                job['skills_by_category'][category] = matching

    return analyzed_jobs

print("Analysis functions loaded\n")


# In[9]:


# --------------- Skill Gap Analysis

def perform_skill_gap_analysis(student_profile: Dict, analyzed_jobs: List[Dict]) -> Dict:
    """Compare student skills with job requirements and identify gaps"""

    print("SKILL GAP ANALYSIS")

    # NEW: More flexible student skill extraction
    student_skills = set()

    # Extract from all skill-related fields in profile
    skill_fields = ['technical_skills', 'soft_skills', 'tools', 'software',
                    'certifications', 'languages', 'frameworks']

    for field in skill_fields:
        if field in student_profile:
            skills_data = student_profile[field]
            if isinstance(skills_data, list):
                for skill in skills_data:
                    student_skills.add(skill.lower().strip())
            elif isinstance(skills_data, str):
                student_skills.add(skills_data.lower().strip())

    # Also extract skills mentioned in projects/experience descriptions
    if 'projects' in student_profile:
        for project in student_profile['projects']:
            if isinstance(project, dict) and 'technologies' in project:
                for tech in project['technologies']:
                    student_skills.add(tech.lower().strip())
            elif isinstance(project, str):
                # Simple extraction from project descriptions
                words = project.lower().split()
                student_skills.update(words)

    print(f"Student has {len(student_skills)} skills")

    # Rest remains the same...
    all_required_skills = Counter()
    job_matches = []

    for job in analyzed_jobs:
        required = set(job["required_skills"])
        all_required_skills.update(required)

        # Calculate match percentage
        matched_skills = student_skills.intersection(required)
        missing_skills = required - student_skills

        if len(required) > 0:
            match_percentage = (len(matched_skills) / len(required)) * 100
        else:
            match_percentage = 0

        job_matches.append({
            "position": job["position"],
            "employer": job["employer"],
            "ref": job["ref"],
            "url": job["url"],
            "match_percentage": round(match_percentage, 2),
            "matched_skills": list(matched_skills),
            "missing_skills": list(missing_skills),
            "total_required": len(required),
            "experience_years": job["experience_years"]
        })

    # Sort jobs by match percentage
    job_matches.sort(key=lambda x: x["match_percentage"], reverse=True)

    # Identify most common missing skills
    all_missing = Counter()
    for match in job_matches:
        all_missing.update(match["missing_skills"])

    # Remove skills student already has
    for skill in student_skills:
        if skill in all_missing:
            del all_missing[skill]

    top_missing_skills = all_missing.most_common(15)

    # Calculate overall readiness
    total_jobs = len(job_matches)
    highly_qualified = len([j for j in job_matches if j["match_percentage"] >= 70])
    moderately_qualified = len([j for j in job_matches if 40 <= j["match_percentage"] < 70])
    needs_improvement = len([j for j in job_matches if j["match_percentage"] < 40])

    gap_analysis = {
        "student_name": student_profile.get("name", "Student"),
        "student_skills_count": len(student_skills),
        "total_jobs_analyzed": total_jobs,
        "job_matches": job_matches,
        "top_missing_skills": [{"skill": skill, "frequency": count} for skill, count in top_missing_skills],
        "most_required_skills": [{"skill": skill, "frequency": count} for skill, count in all_required_skills.most_common(20)],
        "readiness_summary": {
            "highly_qualified": highly_qualified,
            "moderately_qualified": moderately_qualified,
            "needs_improvement": needs_improvement,
            "average_match": round(sum(j["match_percentage"] for j in job_matches) / total_jobs if total_jobs > 0 else 0, 2)
        }
    }

    return gap_analysis

print("Gap analysis functions loaded\n")


# In[10]:


# ------------------Career Opportunity Prediction

def predict_career_opportunities(gap_analysis: Dict, student_profile: Dict) -> Dict:
    """Predict career opportunities and provide recommendations"""

    print("CAREER OPPORTUNITY PREDICTION")

    job_matches = gap_analysis["job_matches"]
    top_missing = gap_analysis["top_missing_skills"]

    # Categorize opportunities
    immediate_opportunities = [j for j in job_matches if j["match_percentage"] >= 70]
    short_term_opportunities = [j for j in job_matches if 50 <= j["match_percentage"] < 70]
    long_term_opportunities = [j for j in job_matches if j["match_percentage"] < 50]

    # Priority skills to learn (most impactful)
    priority_skills = []
    for skill_data in top_missing[:10]:
        skill = skill_data["skill"]
        freq = skill_data["frequency"]

        # Calculate impact (how many jobs would become more accessible)
        jobs_unlocked = len([j for j in job_matches if skill in j["missing_skills"] and j["match_percentage"] >= 60])

        priority_skills.append({
            "skill": skill,
            "frequency": freq,
            "impact_score": jobs_unlocked,
            "priority": "High" if jobs_unlocked >= 3 else "Medium" if jobs_unlocked >= 1 else "Low"
        })

    priority_skills.sort(key=lambda x: (x["impact_score"], x["frequency"]), reverse=True)

    # Learning path recommendation
    learning_path = {
        "immediate_focus": [s for s in priority_skills if s["priority"] == "High"][:5],
        "next_steps": [s for s in priority_skills if s["priority"] == "Medium"][:5],
        "long_term": [s for s in priority_skills if s["priority"] == "Low"][:5]
    }

    # Career growth timeline
    timeline = {
        "0-3_months": {
            "focus": "Apply to immediate opportunities while learning 2-3 high-priority skills",
            "opportunities": len(immediate_opportunities),
            "recommended_skills": [s["skill"] for s in learning_path["immediate_focus"][:3]]
        },
        "3-6_months": {
            "focus": "Expand skill set with medium-priority skills, apply to short-term opportunities",
            "opportunities": len(short_term_opportunities),
            "recommended_skills": [s["skill"] for s in learning_path["next_steps"][:3]]
        },
        "6-12_months": {
            "focus": "Master advanced skills, qualify for long-term opportunities",
            "opportunities": len(long_term_opportunities),
            "recommended_skills": [s["skill"] for s in learning_path["long_term"][:3]]
        }
    }

    prediction = {
        "immediate_opportunities": immediate_opportunities[:10],
        "short_term_opportunities": short_term_opportunities[:10],
        "priority_skills": priority_skills[:10],
        "learning_path": learning_path,
        "career_timeline": timeline,
        "recommendations": generate_recommendations(gap_analysis, student_profile)
    }

    return prediction

def generate_recommendations(gap_analysis: Dict, student_profile: Dict) -> List[str]:
    """Generate personalized career recommendations"""
    recommendations = []

    avg_match = gap_analysis["readiness_summary"]["average_match"]
    highly_qualified = gap_analysis["readiness_summary"]["highly_qualified"]

    if avg_match >= 65:
        recommendations.append("You're well-positioned for many software engineer roles! Focus on applying to jobs with 70%+ match.")
    elif avg_match >= 50:
        recommendations.append("You have a solid foundation. Learning 3-5 key skills will significantly improve your opportunities.")
    else:
        recommendations.append("Focus on building foundational skills. Consider internships or entry-level positions to gain experience.")

    if highly_qualified > 0:
        recommendations.append(f"You qualify for {highly_qualified} positions right now. Start applying!")

    top_missing = gap_analysis["top_missing_skills"][:3]
    if top_missing:
        skills_str = ", ".join([s["skill"].title() for s in top_missing])
        recommendations.append(f"Priority skills to learn: {skills_str}")

    if student_profile.get("experience"):
        recommendations.append("Highlight your internship experience in applications.")
    else:
        recommendations.append("Consider internships or freelance projects to gain practical experience.")

    return recommendations

print("Prediction functions loaded\n")


# In[11]:


# ----------------------- Generate Reports

def generate_reports(
    gap_analysis: Dict,
    predictions: Dict,
    output_folder: str,
    write_local: bool = True,
    storage_prefix: str | None = None,
) -> Dict[str, str | None]:
    """Generate comprehensive reports in JSON and text format"""

    print("GENERATING REPORTS")

    # Save detailed JSON report
    full_report = {
        "gap_analysis": gap_analysis,
        "career_predictions": predictions,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    report_path = None
    report_url = None
    if write_local:
        report_path = os.path.join(output_folder, "career_analysis_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        print(f"JSON report saved: {report_path}")
    else:
        remote = _analysis_path(storage_prefix, "career_analysis_report.json")
        if remote:
            report_url = _upload_json_to_storage(full_report, remote)

    # Generate human-readable text report
    text_report = []
    text_report.append("="*50)
    text_report.append("AI-BASED CAREER GROWTH SYSTEM - ANALYSIS REPORT")
    text_report.append("="*50)
    text_report.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    text_report.append(f"Student: {gap_analysis['student_name']}\n")

    # Summary
    text_report.append("\n" + "-"*50)
    text_report.append("EXECUTIVE SUMMARY")
    text_report.append("-"*50)
    summary = gap_analysis["readiness_summary"]
    text_report.append(f"Total Jobs Analyzed: {gap_analysis['total_jobs_analyzed']}")
    text_report.append(f"Average Match Score: {summary['average_match']}%")
    text_report.append(f"Highly Qualified Jobs: {summary['highly_qualified']}")
    text_report.append(f"Moderately Qualified Jobs: {summary['moderately_qualified']}")
    text_report.append(f"Needs Improvement Jobs: {summary['needs_improvement']}")

    # Top Opportunities
    text_report.append("\n" + "-"*50)
    text_report.append("TOP 10 MATCHING OPPORTUNITIES")
    text_report.append("-"*50)
    for i, job in enumerate(gap_analysis["job_matches"][:10], 1):
        text_report.append(f"\n{i}. {job['position']} at {job['employer']}")
        text_report.append(f"   Match: {job['match_percentage']}% | Ref: {job['ref']}")
        text_report.append(f"   Matched Skills: {', '.join(job['matched_skills'][:5])}")
        if job['missing_skills']:
            text_report.append(f"   Missing Skills: {', '.join(job['missing_skills'][:5])}")

    # Skill Gaps
    text_report.append("\n" + "-"*50)
    text_report.append("TOP 15 SKILL GAPS")
    text_report.append("-"*50)
    for i, skill_data in enumerate(gap_analysis["top_missing_skills"], 1):
        text_report.append(f"{i:2d}. {skill_data['skill'].title():30s} (Required in {skill_data['frequency']} jobs)")

    # Priority Skills
    text_report.append("\n" + "-"*50)
    text_report.append("PRIORITY SKILLS TO LEARN")
    text_report.append("-"*50)
    for i, skill in enumerate(predictions["priority_skills"][:10], 1):
        text_report.append(f"{i:2d}. {skill['skill'].title():30s} Priority: {skill['priority']:6s} | Impact: {skill['impact_score']} jobs")

    # Learning Path
    text_report.append("\n" + "-"*50)
    text_report.append("RECOMMENDED LEARNING PATH")
    text_report.append("-"*50)
    text_report.append("\nImmediate Focus (0-3 months):")
    for skill in predictions["learning_path"]["immediate_focus"]:
        text_report.append(f"  - {skill['skill'].title()}")

    text_report.append("\nNext Steps (3-6 months):")
    for skill in predictions["learning_path"]["next_steps"]:
        text_report.append(f"  - {skill['skill'].title()}")

    text_report.append("\nLong-term Goals (6-12 months):")
    for skill in predictions["learning_path"]["long_term"]:
        text_report.append(f"  - {skill['skill'].title()}")

    # Career Timeline
    text_report.append("\n" + "-"*50)
    text_report.append("CAREER GROWTH TIMELINE")
    text_report.append("-"*50)
    for period, data in predictions["career_timeline"].items():
        text_report.append(f"\n{period.replace('_', '-').upper()}:")
        text_report.append(f"  Focus: {data['focus']}")
        text_report.append(f"  Opportunities: {data['opportunities']} jobs")
        text_report.append(f"  Skills: {', '.join(data['recommended_skills'])}")

    # Recommendations
    text_report.append("\n" + "-"*50)
    text_report.append("PERSONALIZED RECOMMENDATIONS")
    text_report.append("-"*50)
    for rec in predictions["recommendations"]:
        text_report.append(f"\n{rec}")

    text_report.append("\n" + "="*50)

    # Save text report
    text_report_path = None
    text_report_url = None
    if write_local:
        text_report_path = os.path.join(output_folder, "career_analysis_report.txt")
        with open(text_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(text_report))
        print(f"Text report saved: {text_report_path}")
    else:
        remote = _analysis_path(storage_prefix, "career_analysis_report.txt")
        if remote:
            text_report_url = _upload_text_to_storage("\n".join(text_report), remote)

    return {
        "report_path": report_path,
        "text_report_path": text_report_path,
        "report_url": report_url,
        "text_report_url": text_report_url,
    }

print("Report generation functions loaded\n")


# In[12]:


# ---------------- Main Execution Pipeline

def run_analysis(
    keyword: str,
    student_profile: Dict,
    output_folder: str = "topjobs_ads",
    generate_reports_flag: bool = False,
    write_local: bool = True,
    storage_prefix: str | None = None,
) -> Dict:
    """Run analysis pipeline and return structured results."""
    print("AI-BASED CAREER GROWTH SYSTEM - STARTING ANALYSIS")

    start_time = time.time()

    # Step 1: Scrape job ads
    print("STEP 1: Scraping job advertisements...")
    print("-" * 50)
    metadata = scrape_topjobs(keyword, output_folder, write_local=write_local, storage_prefix=storage_prefix)

    if not metadata:
        print(" No jobs found. Exiting.")
        return {
            "metadata": [],
            "analyzed_jobs": [],
            "gap_analysis": None,
            "predictions": None
        }

    storage_files: Dict[str, str | None] = {}

    if write_local:
        os.makedirs(output_folder, exist_ok=True)
        metadata_path = os.path.join(output_folder, "scraped_jobs.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"\n Scraped {len(metadata)} jobs")
    else:
        remote = _analysis_path(storage_prefix, "scraped_jobs.json")
        if remote:
            storage_files["scraped_jobs.json"] = _upload_json_to_storage(metadata, remote)
        print(f"\n Scraped {len(metadata)} jobs")

    # Step 2: Analyze job requirements
    analyzed_jobs = analyze_job_requirements(
        metadata,
        output_folder,
        write_local=write_local,
        storage_prefix=storage_prefix,
    )

    if write_local:
        analyzed_path = os.path.join(output_folder, "analyzed_jobs.json")
        with open(analyzed_path, "w", encoding="utf-8") as f:
            json.dump(analyzed_jobs, f, indent=2, ensure_ascii=False)
        print(f"Analysis complete. Results saved to {analyzed_path}")
    else:
        remote = _analysis_path(storage_prefix, "analyzed_jobs.json")
        if remote:
            storage_files["analyzed_jobs.json"] = _upload_json_to_storage(analyzed_jobs, remote)
        print("Analysis complete. Results uploaded to storage.")

    # Step 3: Perform skill gap analysis
    gap_analysis = perform_skill_gap_analysis(student_profile, analyzed_jobs)

    # Step 4: Predict career opportunities
    predictions = predict_career_opportunities(gap_analysis, student_profile)

    reports = None
    if generate_reports_flag:
        reports = generate_reports(
            gap_analysis,
            predictions,
            output_folder,
            write_local=write_local,
            storage_prefix=storage_prefix,
        )

    elapsed = time.time() - start_time
    print(f"\n Analysis complete in {elapsed:.2f} seconds!")

    return {
        "metadata": metadata,
        "analyzed_jobs": analyzed_jobs,
        "gap_analysis": gap_analysis,
        "predictions": predictions,
        "reports": reports,
        "storage": {
            "prefix": storage_prefix,
            "files": storage_files,
        },
    }


def run_analysis_from_metadata(
    metadata: List[Dict],
    student_profile: Dict,
    output_folder: str = "topjobs_ads",
    generate_reports_flag: bool = False,
    write_local: bool = True,
    storage_prefix: str | None = None,
) -> Dict:
    """Run analysis pipeline using existing scraped metadata."""
    print("AI-BASED CAREER GROWTH SYSTEM - STARTING ANALYSIS (cached metadata)")

    start_time = time.time()
    if not metadata:
        print(" No cached jobs found. Exiting.")
        return {
            "metadata": [],
            "analyzed_jobs": [],
            "gap_analysis": None,
            "predictions": None,
            "reports": None,
            "storage": {"prefix": storage_prefix, "files": {}},
        }

    storage_files: Dict[str, str | None] = {}
    if write_local:
        os.makedirs(output_folder, exist_ok=True)
        metadata_path = os.path.join(output_folder, "scraped_jobs.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"\n Using cached metadata ({len(metadata)} jobs)")
    else:
        remote = _analysis_path(storage_prefix, "scraped_jobs.json")
        if remote:
            storage_files["scraped_jobs.json"] = _upload_json_to_storage(metadata, remote)
        print(f"\n Using cached metadata ({len(metadata)} jobs)")

    analyzed_jobs = analyze_job_requirements(
        metadata,
        output_folder,
        write_local=write_local,
        storage_prefix=storage_prefix,
    )

    if write_local:
        analyzed_path = os.path.join(output_folder, "analyzed_jobs.json")
        with open(analyzed_path, "w", encoding="utf-8") as f:
            json.dump(analyzed_jobs, f, indent=2, ensure_ascii=False)
        print(f"Analysis complete. Results saved to {analyzed_path}")
    else:
        remote = _analysis_path(storage_prefix, "analyzed_jobs.json")
        if remote:
            storage_files["analyzed_jobs.json"] = _upload_json_to_storage(analyzed_jobs, remote)
        print("Analysis complete. Results uploaded to storage.")

    gap_analysis = perform_skill_gap_analysis(student_profile, analyzed_jobs)
    predictions = predict_career_opportunities(gap_analysis, student_profile)

    reports = None
    if generate_reports_flag:
        reports = generate_reports(
            gap_analysis,
            predictions,
            output_folder,
            write_local=write_local,
            storage_prefix=storage_prefix,
        )

    elapsed = time.time() - start_time
    print(f"\n Analysis complete in {elapsed:.2f} seconds!")

    return {
        "metadata": metadata,
        "analyzed_jobs": analyzed_jobs,
        "gap_analysis": gap_analysis,
        "predictions": predictions,
        "reports": reports,
        "storage": {
            "prefix": storage_prefix,
            "files": storage_files,
        },
    }


def main() -> Dict:
    """Main execution pipeline for CLI usage."""
    return run_analysis(KEYWORD, STUDENT_PROFILE, OUTPUT_FOLDER, generate_reports_flag=True)

print("Main pipeline loaded\n")


# In[13]:


if __name__ == "__main__":
    notebook_setup()
    main()
