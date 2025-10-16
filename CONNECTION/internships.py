import requests
import json
import logging
import spacy
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Alternative internship sources
ALTERNATIVE_INTERNSHIPS = {
    "Internshala": "https://internshala.com/internships/",
    "Indeed": "https://www.indeed.com/q-Internship-jobs.html",
    "Glassdoor": "https://www.glassdoor.com/Job/internship-jobs-SRCH_KO0,10.htm",
}

# Fetch LinkedIn Internships
def fetch_linkedin_internships(job_role):
    linkedin_url = f"https://www.linkedin.com/jobs/search/?keywords={job_role.replace(' ', '%20')}&location=Worldwide"

    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-blink-features=AutomationControlled")  # Avoid detection
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")  # User-Agent

    # Auto-install WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        driver.get(linkedin_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "base-search-card__info")))

        soup = BeautifulSoup(driver.page_source, "html.parser")
        job_cards = soup.find_all("div", class_="base-search-card__info")

        internships = []
        for job in job_cards[:5]:
            title = job.find("h3").text.strip() if job.find("h3") else "No Title"
            link_tag = job.find("a", class_="job-card-list__title")
            link = link_tag["href"] if link_tag else linkedin_url
            internships.append({"title": title, "link": link})

        if not internships:
            logging.warning("No LinkedIn internships found. Providing alternative.")
            internships = [{"title": "Check Indeed for internships", "link": ALTERNATIVE_INTERNSHIPS["Indeed"]}]

    except Exception as e:
        logging.error(f"LinkedIn scraping error: {e}")
        internships = [{"title": "Check Indeed for internships", "link": ALTERNATIVE_INTERNSHIPS["Indeed"]}]
    finally:
        driver.quit()

    return internships

# Fetch Data
def recommend_resources(job_role):
    results = {}
    results["LinkedIn Internships"] = fetch_linkedin_internships(job_role)
    return results
