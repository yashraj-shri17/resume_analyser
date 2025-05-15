import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# ✅ Make sure 'model' folder exists
os.makedirs("model", exist_ok=True)

# ✅ List of 50+ Tech Roles
roles = [
    "Software Engineer", "Data Analyst", "Machine Learning Engineer", "Frontend Developer", "Backend Developer",
    "DevOps Engineer", "Data Scientist", "Cloud Engineer", "System Administrator", "QA Tester",
    "Full Stack Developer", "Database Administrator", "Cybersecurity Analyst", "Mobile App Developer",
    "AI Engineer", "Big Data Engineer", "Computer Vision Engineer", "NLP Engineer", "Embedded Systems Engineer",
    "Network Engineer", "Blockchain Developer", "IoT Engineer", "Site Reliability Engineer", "Security Engineer",
    "Game Developer", "AR/VR Developer", "Data Engineer", "UX Designer", "UI Developer",
    "Test Automation Engineer", "Solutions Architect", "Cloud Solutions Architect", "Product Manager (Tech)",
    "BI Developer", "ETL Developer", "Information Security Analyst", "Tech Support Engineer", "Scrum Master",
    "Cloud Security Engineer", "Web Developer", "React Developer", "Angular Developer", "Java Developer",
    "Python Developer", "Node.js Developer", "C++ Developer", "PHP Developer", "Ruby on Rails Developer",
    "Android Developer", "iOS Developer", "Technical Writer", "CRM Developer", "SAP Consultant"
]

# ✅ Fetch job descriptions for all roles
def fetch_jobs_for_all_roles():
    all_data = []
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": "e811b86367mshbd109c23794ffaep131010jsn34683f8ccb67",
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    for role in roles:
        print(f"Fetching jobs for: {role}")
        querystring = {"query": role, "page": "1", "num_pages": "2"}  # reduce pages if limit hit
        try:
            response = requests.get(url, headers=headers, params=querystring)
            jobs = response.json().get('data', [])
            for job in jobs:
                desc = job.get('job_description', '')
                skills = ' '.join(job.get("job_required_skills", []))
                degree = ' '.join(job.get("job_highlights", {}).get("Qualifications", []))
                all_data.append({
                    "text": f"{desc} {skills} {degree}",
                    "label": role
                })
        except Exception as e:
            print(f"Failed for {role}: {e}")
    
    return pd.DataFrame(all_data)

# ✅ Train and save the model
def train_model(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['text'])
    y = df['label']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    with open("model/job_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

# ✅ Run it all
df = fetch_jobs_for_all_roles()
print(f"Total records fetched: {len(df)}")
train_model(df)
print("✅ Model trained and saved successfully!")
