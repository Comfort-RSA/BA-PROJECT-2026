"""
app.py
 
Step 6  : the website.
 
Ties together everything already built:
    - stream_model.pkl / label_encoder.pkl  
    - nlp_chatbot.py                        
    - combine_results.py                    
    - schools.json                         
Pages:
    /            Home - introduction
    /school      Optional: learner picks their high school
    /assessment  Learner enters their 9 Grade 9 subject marks
    /chatbot     Learner answers interest/career questions
    /recommendation  Shows the final combined recommendation
    /about       Explains the AI/ML/NLP behind the tool
 
Run:
    python app.py
Then open: http://127.0.0.1:5000 in the browser.
"""
 
import json
from pathlib import Path
 
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
 
from combine_results import load_model, recommend, FEATURE_COLS
 
app = Flask(__name__)
app.secret_key = "stream-advisor-dev-key"  # fine for local student project use
 
MODEL, ENCODER = load_model()
 
# ---------------------------------------------------------------------------
# Project contact details — CHANGE THESE TWO VALUES to update the footer
# site-wide (they're used in base.html via the context processor below).
# ---------------------------------------------------------------------------
PROJECT_EMAIL = "streamadvisor.project@example.com"   # placeholder — replace with your real project email
PROJECT_URL = "https://github.com/Comfort-RSA/BA-PROJECT-2026"  # placeholder — replace with your real repo/project link
 
BASE_DIR = Path(__file__).resolve().parent
SCHOOLS_PATH = BASE_DIR / "schools.json"
 
with open(SCHOOLS_PATH, encoding="utf-8") as f:
    SCHOOLS_DATA = json.load(f)
SCHOOLS = SCHOOLS_DATA["schools"]
SCHOOLS_BY_NAME = {s["name"]: s for s in SCHOOLS}
 
 
@app.context_processor
def inject_site_globals():
    """Makes these available in every template without passing them manually."""
    return {
        "project_email": PROJECT_EMAIL,
        "project_url": PROJECT_URL,
    }
 
# (form field name / dataset column, label shown to the learner)
# The field names MUST match FEATURE_COLS in train_model.py.
FIELDS = [
    ("Mathematics", "Mathematics"),
    ("Life_Orientation", "Life Orientation"),
    ("First_Additional_Language", "First Additional Language"),
    ("Social_Science", "Social Science"),
    ("EMS", "Economic & Management Sciences (EMS)"),
    ("Home_Language", "Home Language"),
    ("Creative_Arts", "Creative Arts"),
    ("Natural_Science", "Natural Science"),
    ("Technology", "Technology"),
]
 
# Safety check: if the form and the trained model ever disagree, fail loudly
# at start-up instead of giving a confusing error later.
assert [f for f, _ in FIELDS] == FEATURE_COLS, "FIELDS in app.py must match FEATURE_COLS in train_model.py"
 
 
@app.route("/")
def home():
    return render_template("home.html")
 
 
@app.route("/school", methods=["GET", "POST"])
def select_school():
    """Step 0 (optional): learner searches for and selects their high school.
    The selected school's offered streams are stored in the session and
    carried through to the assessment + recommendation pages, so the final
    result can note whether the recommended stream is offered at that
    learner's school."""
    if request.method == "POST":
        school_name = request.form.get("school_name", "").strip()
        school = SCHOOLS_BY_NAME.get(school_name)
        if not school:
            return render_template(
                "school.html",
                schools=SCHOOLS,
                error="Please select a valid school from the list.",
            )
        session["school"] = school
        return redirect(url_for("assessment"))
 
    return render_template("school.html", schools=SCHOOLS, error=None)
 
 
@app.route("/school/skip")
def skip_school():
    session.pop("school", None)
    return redirect(url_for("assessment"))
 
 
@app.route("/api/schools")
def api_schools():
    """Small JSON endpoint the school-search widget fetches from."""
    return jsonify(SCHOOLS)
 
 
@app.route("/assessment", methods=["GET", "POST"])
def assessment():
    if request.method == "POST":
        marks = {}
        error = None
        for field, label in FIELDS:
            raw = request.form.get(field, "").strip()
            try:
                value = float(raw)
            except ValueError:
                error = "Please enter a valid number for every subject."
                break
            if not (0 <= value <= 100):
                error = f"{label} must be between 0 and 100."
                break
            marks[field] = value
 
        if error:
            return render_template("assessment.html", fields=FIELDS, error=error, school=session.get("school"))
 
        session["marks"] = marks
        session.pop("result", None)   # clear any older recommendation
        return redirect(url_for("chatbot"))
 
    return render_template("assessment.html", fields=FIELDS, error=None, school=session.get("school"))
 
 
@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if "marks" not in session:
        return redirect(url_for("assessment"))
 
    if request.method == "POST":
        interest_text = request.form.get("interest_text", "").strip()
        career_text = request.form.get("career_text", "").strip()
 
        if not interest_text or not career_text:
            return render_template("chatbot.html", error="Please answer both questions.", school=session.get("school"))
 
        combined_text = f"{interest_text} I am interested in {career_text}."
        marks = session["marks"]
 
        result = recommend(marks, combined_text, MODEL, ENCODER, verbose=False)
        session["result"] = result
        return redirect(url_for("recommendation"))
 
    return render_template("chatbot.html", error=None, school=session.get("school"))
 
 
@app.route("/recommendation")
def recommendation():
    result = session.get("result")
    if not result:
        return redirect(url_for("assessment"))
 
    school = session.get("school")
    school_offers_stream = None
    if school and school.get("streams"):
        # streams is only populated for hand-verified schools - an empty/missing
        # list means "not verified", not "doesn't offer this stream", so
        # school_offers_stream stays None (not False) in that case.
        school_offers_stream = result["recommended_stream"] in school["streams"]
 
    return render_template(
        "recommendation.html",
        result=result,
        school=school,
        school_offers_stream=school_offers_stream,
    )
 
 
@app.route("/about")
def about():
    return render_template("about.html")
 
 
if __name__ == "__main__":
    app.run(debug=True)