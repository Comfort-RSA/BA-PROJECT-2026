"""
train_model.py
 
Step 2-3 of your project workflow: Train and test the Machine Learning
models (Decision Tree + Random Forest) that predict Recommended_Stream
from the learner's Grade 9 subject marks.
 
Uses: Student_performance_data_.csv
      (2392 learners, 4 streams, split column "Dataset Split" = Train / Test)
 
What changed for this dataset (good to mention in your presentation):
 
1. FILE FORMAT: this dataset is a .csv file, not .xlsx, so it's read with
   pd.read_csv() instead of pd.read_excel() - no sheet name needed.
 
2. ROW COUNT: 2,392 learners (was 600), with the same 60/40 Train/Test
   split (1435 Train / 957 Test), stratified by Recommended Stream.
 
3. ID COLUMN: "StudentID" (was "Learner_ID").
 
4. FEATURES: the same 9 raw Grade 9 subject marks as before:
   Mathematics, Life Orientation, First Additional Language, Social Science,
   EMS, Home Language, Creative Arts, Natural Science, Technology.
 
5. TARGET: Recommended Stream, 4 classes:
   Science, Commerce, Humanities, Technical/Vocational.
 
6. NEW COLUMNS IN THIS DATASET - checked and NOT used as model inputs:
   - Age, Gender, StudyTimeWeekly, Absences, Tutoring, ParentalSupport,
     Extracurricular: their averages are almost identical across all 4
     streams, so they carry no information about which stream fits a
     learner. Using them would just add noise.
   - Favourite Subjects: a free-text summary of the same 9 marks - it
     doesn't add anything the marks don't already say.
   - Confidence Score % : still ~87% for every stream (not informative).
   - Average Mark, Career Goal, Chatbot Interest Response : same reasons
     as before - Average Mark is just a formula over the 9 marks, and the
     two text columns are handled by the NLP module (nlp_chatbot.py), not
     this model.
 
Run:
    python train_model.py
"""
 
from pathlib import Path
 
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
 
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Student_performance_data_.csv"
MODEL_PATH = BASE_DIR / "stream_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
 
# Column names have spaces replaced by underscores when the data is loaded
# ("Life Orientation" -> "Life_Orientation"), so they are safe to use as
# form-field names on the website.
FEATURE_COLS = [
    "Mathematics",
    "Life_Orientation",
    "First_Additional_Language",
    "Social_Science",
    "EMS",
    "Home_Language",
    "Creative_Arts",
    "Natural_Science",
    "Technology",
]
TARGET_COL = "Recommended_Stream"
ID_COL = "StudentID"
 
 
def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.replace(" ", "_")
 
    print("=== Step 1: Data cleaning checks ===")
    print(f"Rows: {len(df)}")
 
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL, "Dataset_Split", ID_COL] if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"ERROR: these columns are missing from the dataset: {missing_cols}")
 
    missing = df[FEATURE_COLS + [TARGET_COL]].isnull().sum()
    missing = missing[missing > 0]
    print(f"Missing values in feature/target columns: {'None found' if missing.empty else dict(missing)}")
    print(f"Duplicate {ID_COL}s: {df[ID_COL].duplicated().sum()}")
 
    out_of_range = ((df[FEATURE_COLS] < 0) | (df[FEATURE_COLS] > 100)).sum().sum()
    print(f"Marks outside 0-100: {out_of_range}")
    print(f"Mark columns are numeric: {all(pd.api.types.is_numeric_dtype(df[c]) for c in FEATURE_COLS)}")
    print(f"Stream counts:\n{df[TARGET_COL].value_counts().to_string()}")
    print()
    return df
 
 
def split_data(df):
    train_df = df[df["Dataset_Split"] == "Train"]
    test_df = df[df["Dataset_Split"] == "Test"]
    print("=== Step 2: Using the dataset's Train/Test split ===")
    print(f"Training rows: {len(train_df)} | Testing rows: {len(test_df)}")
    print(f"({len(train_df) / len(df):.0%}/{len(test_df) / len(df):.0%} split, stratified by Recommended_Stream)")
    print("Stream counts per split:")
    print(pd.crosstab(df[TARGET_COL], df["Dataset_Split"]).to_string())
    print()
    return train_df, test_df
 
 
def train_and_evaluate(train_df, test_df, label_encoder):
    X_train = train_df[FEATURE_COLS]
    y_train = label_encoder.transform(train_df[TARGET_COL])
    X_test = test_df[FEATURE_COLS]
    y_test = label_encoder.transform(test_df[TARGET_COL])
 
    results = {}
 
    for name, model in [
        ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
    ]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
 
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
 
        # 5-fold cross-validation on the TRAINING rows only (test rows stay unseen)
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
 
        print(f"=== {name} results (on the {len(test_df)} unseen test learners) ===")
        print(f"Accuracy:  {acc:.2%}")
        print(f"Precision: {prec:.2%}")
        print(f"Recall:    {rec:.2%}")
        print(f"F1-score:  {f1:.2%}")
        print(f"5-fold cross-validation accuracy (training data): {cv.mean():.2%} (+/- {cv.std():.2%})")
        print("Confusion matrix (rows=actual, cols=predicted):")
        print(f"Classes: {list(label_encoder.classes_)}")
        print(cm)
        print()
 
        results[name] = {"model": model, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
 
    return results
 
 
def show_feature_importance(model, name):
    print(f"=== Which subjects matter most to the {name}? ===")
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    for col, val in importance.items():
        print(f"  {col:<28} {val:.1%}")
    print()
 
 
def show_tree_explanation(model, feature_names, label_encoder):
    print("=== Decision Tree logic (first few levels, for your presentation) ===")
    legend = ", ".join(f"class {i} = {c}" for i, c in enumerate(label_encoder.classes_))
    print(f"({legend})")
    print(export_text(model, feature_names=feature_names, max_depth=3))
 
 
def main():
    df = load_and_clean()
    train_df, test_df = split_data(df)
 
    label_encoder = LabelEncoder()
    label_encoder.fit(df[TARGET_COL])
    print(f"Stream classes: {list(label_encoder.classes_)}\n")
 
    results = train_and_evaluate(train_df, test_df, label_encoder)
 
    # Best = highest accuracy (F1 breaks a tie)
    best_name = max(results, key=lambda k: (results[k]["accuracy"], results[k]["f1"]))
    best_model = results[best_name]["model"]
    print(f"Best model: {best_name} (Accuracy: {results[best_name]['accuracy']:.2%})\n")
 
    show_feature_importance(best_model, best_name)
    if "Decision Tree" in results:
        show_tree_explanation(results["Decision Tree"]["model"], FEATURE_COLS, label_encoder)
 
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    print(f"\nSaved model -> {MODEL_PATH.name}")
    print(f"Saved label encoder -> {ENCODER_PATH.name}")
 
    # Example: predict a new learner
    print("\n=== Example prediction for a new learner ===")
    example = pd.DataFrame([{
        "Mathematics": 85, "Life_Orientation": 70, "First_Additional_Language": 60,
        "Social_Science": 55, "EMS": 58, "Home_Language": 65,
        "Creative_Arts": 62, "Natural_Science": 82, "Technology": 78,
    }])
    pred = best_model.predict(example[FEATURE_COLS])
    pred_label = label_encoder.inverse_transform(pred)[0]
    print(f"Input marks: {example.to_dict('records')[0]}")
    print(f"Predicted stream: {pred_label}")
 
 
if __name__ == "__main__":
    main()
