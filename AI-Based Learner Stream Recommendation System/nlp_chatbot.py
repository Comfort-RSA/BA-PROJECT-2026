"""
nlp_chatbot.py

Step 4 : NLP chatbot that reads a learner's
interest/career text and scores how strongly it relates to each of the
4 streams in the dataset:

    Science | Commerce | Humanities | Technical/Vocational

Approach: keyword/phrase matching against a per-stream vocabulary.

    python nlp_chatbot.py - accuracy check on the dataset + interactive chatbot
"""

import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Student_performance_data_.csv"

STREAMS = ["Science", "Commerce", "Humanities", "Technical/Vocational"]

# Lowercase words/phrases. A trailing * = "any word that starts with this".
STREAM_KEYWORDS = {
    "Science": [
        # school subjects / academic interests
        "maths", "math", "mathematics", "numbers", "science", "scientific",
        "biology", "chemistry", "physics", "space", "astronomy",
        "experiment*", "discover*", "curious", "research*", "concepts",
        "problem*", "solv*", "how things work", "human body",
        "technology", "computer*", "coding", "programming", "data",
        "medicine", "medical", "health", "nurse*", "animals",
        # careers in the dataset (business/teacher/psychologist/electrician etc.
        # are already covered by the word-stems in their own stream's list)
        "doctor", "pharmacist", "vet", "veterinar*", "pilot", "architect",
        "engineer*", "data scientist", "software developer", "developer",
    ],
    "Commerce": [
        "business*", "money", "financ*", "accounting", "accountant",
        "entrepreneur*", "manag*", "trade", "trading", "marketing",
        "marketer", "bank*", "budget*", "econom*", "compan*", "profit",
        "sales", "invest*", "ems", "stock*", "own business",
        # careers in the dataset
        "financial analyst",
    ],
    "Humanities": [
        "writ*", "read*", "history", "language*", "debat*", "social",
        "societ*", "cultur*", "law", "lawyer", "journalis*", "teach*",
        "psycholog*", "art", "arts", "creative*", "stor*", "politic*",
        "philosoph*", "communit*", "helping", "people", "current affairs",
        "discuss*", "world",
        # careers in the dataset
        "social worker", "diplomat*",
    ],
    "Technical/Vocational": [
        "hands-on", "hands", "practical", "mechanic*", "electric*",
        "plumb*", "carpentry", "construction", "workshop", "repair*",
        "fix*", "machine*", "machinery", "technician*", "technical",
        "welding", "tools", "tool", "apart", "back together", "cars", "engines",
        # careers in the dataset
        "artisan", "boilermaker", "fitter and turner", "civil technician",
    ],
}


def _compile(keyword):
    """Turn a vocabulary entry into a whole-word (or word-stem) regex."""
    kw = keyword.strip().lower()
    if kw.endswith("*"):
        return re.compile(r"\b" + re.escape(kw[:-1]) + r"\w*")
    return re.compile(r"\b" + re.escape(kw) + r"\b")


# Pre-compile once so scoring is fast.
_PATTERNS = {
    stream: [(kw.rstrip("*"), _compile(kw)) for kw in keywords]
    for stream, keywords in STREAM_KEYWORDS.items()
}


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)   # keep hyphens (hands-on)
    text = re.sub(r"\s+", " ", text)
    return text


def explain_matches(text):
    """Which vocabulary entries matched, per stream. Useful for your presentation."""
    cleaned = clean_text(text)
    explanation = {}
    for stream, patterns in _PATTERNS.items():
        matched = [kw for kw, rx in patterns if rx.search(cleaned)]
        if matched:
            explanation[stream] = matched
    return explanation


def score_text_against_streams(text):
    """
    Returns {stream: score_percentage} based on keyword matches.

    Score = matched keywords for that stream, normalised so all stream
    scores add up to 100%. If nothing matches at all, returns an even
    split (i.e. "no signal either way").
    """
    matches = explain_matches(text)
    counts = {s: len(matches.get(s, [])) for s in STREAMS}
    total = sum(counts.values())

    if total == 0:
        even = round(100 / len(STREAMS), 1)
        return {s: even for s in STREAMS}

    return {s: round(c / total * 100, 1) for s, c in counts.items()}


def analyse_text(text):
    """
    One call that returns everything the rest of the project needs:
        scores       {stream: %}
        top_stream   best NLP match (None if there was no signal)
        matches      {stream: [keywords]}
        has_signal   False if no keyword matched at all
    """
    matches = explain_matches(text)
    has_signal = len(matches) > 0
    scores = score_text_against_streams(text)
    top_stream = max(scores, key=scores.get) if has_signal else None
    return {
        "scores": scores,
        "top_stream": top_stream,
        "matches": matches,
        "has_signal": has_signal,
    }


def build_text(interest_text, career_text):
    """The learner's two chatbot answers combined into one piece of text."""
    return f"{interest_text} I am interested in {career_text}."


def demo_on_dataset(n=5):
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.replace(" ", "_")

    print(f"=== NLP scoring demo on {n} learners from the dataset ===\n")
    for _, row in df.head(n).iterrows():
        text = build_text(row["Chatbot_Interest_Response"], row["Career_Goal"])
        result = analyse_text(text)
        top = result["top_stream"]

        print(f"Learner: {row['StudentID']}")
        print(f'  Text: "{text}"')
        print(f"  NLP scores: {result['scores']}")
        print(f"  Top NLP match: {top}")
        print(f"  Matched keywords: {result['matches'].get(top, [])}")
        print(f"  (Dataset's Recommended_Stream: {row['Recommended_Stream']})")
        print()

    # Accuracy of the NLP module on its own (on the unseen Test rows).
    test_df = df[df["Dataset_Split"] == "Test"]
    predictions = [
        analyse_text(build_text(r["Chatbot_Interest_Response"], r["Career_Goal"]))["top_stream"]
        for _, r in test_df.iterrows()
    ]
    correct = sum(p == a for p, a in zip(predictions, test_df["Recommended_Stream"]))
    no_signal = sum(p is None for p in predictions)
    print("=== NLP-only accuracy on the Test rows ===")
    print(f"Correct: {correct}/{len(test_df)} = {correct / len(test_df):.1%}")
    print(f"Rows with no keyword match at all: {no_signal}")
    print("(High because the dataset's chatbot answers are templated - see notes.)")


def interactive_chatbot():
    print("=== AI Stream Recommendation Chatbot (NLP demo) ===")
    print("Answer a couple of questions and I'll show you the NLP-only score.")
    print("(Type 'quit' at any point to exit)\n")

    q1 = input("Chatbot: What subjects do you enjoy most, and why?\nYou: ")
    if q1.strip().lower() == "quit":
        return
    q2 = input("Chatbot: What career are you interested in?\nYou: ")
    if q2.strip().lower() == "quit":
        return

    result = analyse_text(build_text(q1, q2))

    print("\n=== NLP Analysis ===")
    if not result["has_signal"]:
        print("I couldn't pick out clear interest words in your answers.")
        print("Try mentioning subjects, activities or careers you like.")
    else:
        for stream, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
            print(f"  {stream}: {score}%")
        top = result["top_stream"]
        print(f"\nTop match based on your answers: {top}")
        print(f"Keywords that influenced this: {', '.join(result['matches'][top])}")

    print("\n(Note: this is the NLP score only. The final recommendation")
    print("combines this with your Grade 9 marks - see combine_results.py)")


if __name__ == "__main__":
    demo_on_dataset(5)
    print("\n" + "=" * 60 + "\n")
    interactive_chatbot()
