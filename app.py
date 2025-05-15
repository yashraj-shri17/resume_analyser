from flask import Flask, request, render_template
import pickle
from resume_parser import extract_text_from_pdf
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

model = pickle.load(open("model/job_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        resume = request.files["resume"]
        text = extract_text_from_pdf(resume)
        vect_text = vectorizer.transform([text])
        
        # Get probabilities for each class
        probs = model.predict_proba(vect_text)[0]
        classes = model.classes_
        
        # Get indices of top 3 probabilities
        top_n = 5
        top_indices = probs.argsort()[-top_n:][::-1]
        
        # Prepare list of (job_profile, confidence)
        suggestions = [(classes[i], round(probs[i]*100, 2)) for i in top_indices]
        
        # Render results as HTML
        result_html = "<h2>Top Job Profile Suggestions:</h2><ul>"
        for job, conf in suggestions:
            result_html += f"<li>{job}"
        result_html += "</ul>"
        
        return result_html
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
