"""
combine_results.py
combine the ML score (from Grade 9 marks)
with the NLP score (from the chatbot answers) into ONE final recommendation.

    final score = ML_WEIGHT * ML score  +  NLP_WEIGHT * NLP score

Both scores are percentages (0-100) for each of the 4 streams, so they can
be added together directly.

If the NLP module finds no interest keywords at all in the learner's text,
the recommendation falls back to the marks-only (ML) result rather than
letting a meaningless "even split" dilute the confidence.

"""

import joblib
import pandas as pd

from nlp_chatbot import STREAMS, analyse_text, build_text
from train_model import DATA_PATH, ENCODER_PATH, FEATURE_COLS, MODEL_PATH, TARGET_COL

# Easy to change - and easy to explain in your presentation.
ML_WEIGHT = 0.6
NLP_WEIGHT = 0.4

# Nice labels for the subjects.
SUBJECT_LABELS = {
    "Mathematics": "Mathematics",
    "Life_Orientation": "Life Orientation",
    "First_Additional_Language": "First Additional Language",
    "Social_Science": "Social Science",
    "EMS": "EMS",
    "Home_Language": "Home Language",
    "Creative_Arts": "Creative Arts",
    "Natural_Science": "Natural Science",
    "Technology": "Technology",
}

# Subjects that matter most for each stream (based on the subject averages in
# the dataset). Used only to write the "Reason" sentence.
STREAM_SUBJECTS = {
    "Science": ["Mathematics", "Natural_Science", "Technology"],
    "Commerce": ["EMS", "Mathematics"],
    "Humanities": ["Social_Science", "First_Additional_Language", "Home_Language"],
    "Technical/Vocational": ["Technology", "Natural_Science", "Mathematics"],
}

# Example careers per stream (taken from the Career Goal column in the dataset).
STREAM_CAREERS = {
    "Science": ["Engineer", "Doctor", "Software Developer", "Data Scientist"],
    "Commerce": ["Accountant", "Business Manager", "Banker", "Entrepreneur"],
    "Humanities": ["Lawyer", "Teacher", "Journalist", "Psychologist"],
    "Technical/Vocational": ["Electrician", "Mechanic", "Civil Technician", "Artisan"],
}


def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
    except FileNotFoundError:
        raise SystemExit(
            "Model files not found. Run 'py train_model.py' first "
            "to create stream_model.pkl and label_encoder.pkl."
        )
    return model, encoder


def ml_scores(marks, model, encoder):
    """{stream: probability %} from the trained model, using the 9 subject marks."""
    X = pd.DataFrame([marks])[FEATURE_COLS]
    probabilities = model.predict_proba(X)[0]
    stream_names = encoder.inverse_transform(model.classes_)
    return {stream: round(float(p) * 100, 1) for stream, p in zip(stream_names, probabilities)}


def strongest_subjects(marks, n=3):
    ranked = sorted(FEATURE_COLS, key=lambda c: marks[c], reverse=True)[:n]
    return [{"subject": SUBJECT_LABELS[c], "mark": float(marks[c])} for c in ranked]


def build_reason(stream, marks, matched_keywords, nlp_used):
    relevant = sorted(STREAM_SUBJECTS[stream], key=lambda c: marks[c], reverse=True)[:2]
    subjects = ", ".join(SUBJECT_LABELS[c] for c in relevant)
    if nlp_used and matched_keywords:
        return (f"Strong performance in {subjects}, combined with a stated interest in "
                f"{', '.join(matched_keywords[:4])}.")
    return (f"Strong performance in {subjects}. No clear interest keywords were found in "
            f"the chatbot answers, so this is based on your marks.")


def recommend(marks, text, model, encoder, verbose=True):
    """
    marks : {feature column: mark 0-100} for the 9 subjects
    text  : the learner's combined chatbot answers
    Returns a plain dict (safe to store in a Flask session).
    """
    marks = {c: float(marks[c]) for c in FEATURE_COLS}

    ml = ml_scores(marks, model, encoder)
    nlp = analyse_text(text)

    if nlp["has_signal"]:
        ml_w, nlp_w = ML_WEIGHT, NLP_WEIGHT
    else:
        ml_w, nlp_w = 1.0, 0.0   # no interest keywords found -> marks only

    final = {s: round(ml_w * ml[s] + nlp_w * nlp["scores"][s], 1) for s in STREAMS}
    ranking = sorted(final.items(), key=lambda item: -item[1])
    best_stream, best_score = ranking[0]
    runner_up, runner_up_score = ranking[1]
    matched = nlp["matches"].get(best_stream, [])
    reason = build_reason(best_stream, marks, matched, nlp["has_signal"])

    result = {
        "recommended_stream": best_stream,
        "confidence": best_score,
        "fit": best_score,                       # same number, extra names so
        "fit_percentage": best_score,            # older templates keep working
        "reason": reason,
        "careers": STREAM_CAREERS[best_stream],
        "possible_careers": STREAM_CAREERS[best_stream],
        "runner_up": runner_up,
        "runner_up_score": runner_up_score,
        # list of {stream, score}, best first - Flask sessions do not keep dict order
        "ranking": [{"stream": s, "score": v} for s, v in ranking],
        "final_scores": final,
        "combined_scores": final,
        "ml_scores": ml,
        "nlp_scores": nlp["scores"],
        "nlp_used": nlp["has_signal"],
        "nlp_top_stream": nlp["top_stream"],
        "matched_keywords": matched,
        "all_matches": nlp["matches"],
        "strongest_subjects": strongest_subjects(marks),
        "weights": {"ml": ml_w, "nlp": nlp_w},
    }

    if verbose:
        print("\n=== Final Recommendation ===")
        print(f"Recommended Stream: {best_stream}")
        print(f"Fit: {best_score}%")
        print(f"Reason: {reason}")
        print(f"Possible careers: {', '.join(STREAM_CAREERS[best_stream])}")
        print(f"\n(ML scores: {ml})")
        print(f"(NLP scores: {nlp['scores']})")
        print(f"(Combined scores: {final})")
        print(f"(Weights used: ML {ml_w:.0%} / NLP {nlp_w:.0%})")

    return result


def validate(model, encoder):
    """ML-only vs NLP-only vs Combined accuracy on the unseen Test learners."""
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.replace(" ", "_")
    test_df = df[df["Dataset_Split"] == "Test"]
    actual = list(test_df[TARGET_COL])
    n = len(test_df)

    ml_pred = list(encoder.inverse_transform(model.predict(test_df[FEATURE_COLS])))

    nlp_pred, combined_pred = [], []
    for _, row in test_df.iterrows():
        text = build_text(row["Chatbot_Interest_Response"], row["Career_Goal"])
        nlp_pred.append(analyse_text(text)["top_stream"])
        marks = {c: row[c] for c in FEATURE_COLS}
        combined_pred.append(recommend(marks, text, model, encoder, verbose=False)["recommended_stream"])

    def correct(pred):
        return sum(p == a for p, a in zip(pred, actual))

    print("=== Validation: ML-only vs NLP-only vs Combined ===")
    print(f"ML-only accuracy:       {correct(ml_pred)}/{n} = {correct(ml_pred) / n:.1%}")
    print(f"NLP-only accuracy:      {correct(nlp_pred)}/{n} = {correct(nlp_pred) / n:.1%}")
    print(f"Combined accuracy:      {correct(combined_pred)}/{n} = {correct(combined_pred) / n:.1%}")
    print("(NLP is very high because the dataset's chatbot answers are templated.)")


def _ask_mark(label):
    while True:
        raw = input(f"{label} %: ").strip()
        try:
            value = float(raw)
            if 0 <= value <= 100:
                return value
        except ValueError:
            pass
        print("  Please enter a number from 0 to 100.")


def try_it_yourself(model, encoder):
    print("=== Try it yourself ===")
    marks = {c: _ask_mark(SUBJECT_LABELS[c]) for c in FEATURE_COLS}
    interest = input("What subjects do you enjoy most, and why?\nYou: ")
    career = input("What career are you interested in?\nYou: ")
    recommend(marks, build_text(interest, career), model, encoder, verbose=True)


if __name__ == "__main__":
    model, encoder = load_model()

    validate(model, encoder)

    print("\n" + "=" * 60 + "\n")
    print("=== Worked example ===")
    example_marks = {
        "Mathematics": 88, "Life_Orientation": 70, "First_Additional_Language": 60,
        "Social_Science": 55, "EMS": 58, "Home_Language": 65,
        "Creative_Arts": 62, "Natural_Science": 85, "Technology": 78,
    }
    example_text = "I enjoy Maths and Science because I like solving problems and learning how things work."
    print(f"Marks: {example_marks}")
    print(f'Interest text: "{example_text}"')
    recommend(example_marks, example_text, model, encoder, verbose=True)

    print("\n" + "=" * 60 + "\n")
    try_it_yourself(model, encoder)