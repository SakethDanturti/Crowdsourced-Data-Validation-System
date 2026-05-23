import pandas as pd
import numpy as np
import joblib
import re

from collections import defaultdict

from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from email_validator import validate_email, EmailNotValidError
DISPOSABLE_DOMAINS = [
    "tempmail.com",
    "10minutemail.com",
    "mailinator.com",
    "guerrillamail.com",
    "yopmail.com"
]

# =========================================================
# LOAD DATASETS
# =========================================================

benchmark_df = pd.read_csv("benchmark_ranges.csv")

# =========================================================
# LOAD ML MODELS
# =========================================================

salary_model = joblib.load("salary_model.pkl")

encoders = joblib.load("encoder.pkl")

# =========================================================
# VALID LISTS
# =========================================================

VALID_COMPANIES = {
    "google": "google.com",
    "meta": "meta.com",
    "amazon": "amazon.com",
    "microsoft": "microsoft.com",
    "apple": "apple.com",
    "netflix": "netflix.com",
    "tesla": "tesla.com",
    "uber": "uber.com",
    "adobe": "adobe.com",
    "salesforce": "salesforce.com"
}

VALID_LOCATIONS = [
    "bangalore",
    "hyderabad",
    "mumbai",
    "delhi",
    "pune",
    "chennai",
    "san francisco",
    "seattle",
    "new york",
    "mountain view",
    "london",
    "singapore"
]

VALID_LEVELS = [
    "intern",
    "l1",
    "l2",
    "l3",
    "l4",
    "l5",
    "manager",
    "senior manager",
    "director"
]

# =========================================================
# MEMORY TRACKING
# =========================================================

SEEN_SESSIONS = set()

SEEN_EMAILS = defaultdict(int)

SEEN_TEXTS = []

SEEN_COMPENSATIONS = []

# =========================================================
# ANOMALY MODEL
# =========================================================

isolation_model = IsolationForest(
    contamination=0.05,
    random_state=42
)

# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(value):

    return str(value).strip().lower()

# =========================================================
# EMAIL VALIDATION
# =========================================================

def validate_email_address(email):

    try:

        valid = validate_email(email)

        return True, valid.email

    except EmailNotValidError:

        return False, email

# =========================================================
# DISPOSABLE EMAIL CHECK
# =========================================================

def check_disposable_email(email):

    domain = email.split("@")[-1]

    return domain in DISPOSABLE_DOMAINS

# =========================================================
# DUPLICATE DETECTION
# =========================================================

def detect_duplicate_session(session_id):

    if session_id in SEEN_SESSIONS:
        return True

    SEEN_SESSIONS.add(session_id)

    return False

# =========================================================
# SPAM DETECTION
# =========================================================

def detect_email_spam(email):

    SEEN_EMAILS[email] += 1

    return SEEN_EMAILS[email] > 3

# =========================================================
# TEXT FRAUD DETECTION
# =========================================================

def detect_suspicious_keywords(text):

    patterns = [
        "guaranteed",
        "100%",
        "million",
        "fake",
        "easy money",
        "hack"
    ]

    detected = []

    for pattern in patterns:

        if pattern in text:
            detected.append(pattern)

    return detected

# =========================================================
# NLP SIMILARITY DETECTION
# =========================================================

def detect_repeated_claims(current_text):

    global SEEN_TEXTS

    if len(SEEN_TEXTS) == 0:

        SEEN_TEXTS.append(current_text)

        return False, 0

    texts = SEEN_TEXTS + [current_text]

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(texts)

    similarity_matrix = cosine_similarity(vectors)

    latest = similarity_matrix[-1][:-1]

    similarity = max(latest)

    SEEN_TEXTS.append(current_text)

    return similarity > 0.90, similarity

# =========================================================
# ML SALARY PREDICTION
# =========================================================

def predict_salary(data):

    input_df = pd.DataFrame([{
        "company": data["company"],
        "education": data["education"],
        "role": data["role"],
        "level_designation": data["level_designation"],
        "years_of_experience": data["years_of_experience"],
        "location": data["location"],
        "employment_type": data["employment_type"]
    }])

    categorical_cols = [
        "company",
        "education",
        "role",
        "level_designation",
        "location",
        "employment_type"
    ]

    for col in categorical_cols:

        le = encoders[col]

        value = str(input_df[col][0])

        if value not in le.classes_:

            input_df[col] = 0

        else:

            input_df[col] = le.transform([value])

    features = [[
        input_df["company"][0],
        input_df["education"][0],
        input_df["role"][0],
        input_df["level_designation"][0],
        input_df["years_of_experience"][0],
        input_df["location"][0],
        input_df["employment_type"][0]
    ]]

    prediction = salary_model.predict(features)[0]

    return prediction

# =========================================================
# BENCHMARK VALIDATION
# =========================================================

def benchmark_validation(data):

    company = data["company"]

    role = data["role"]

    location = data["location"]

    match = benchmark_df[
        (benchmark_df["company"] == company)
        & (benchmark_df["role"] == role)
        & (benchmark_df["location"] == location)
    ]

    if match.empty:

        return {
            "benchmark_score": 50,
            "benchmark_issue": "No benchmark found"
        }

    min_salary = match.iloc[0]["min_salary"]

    max_salary = match.iloc[0]["max_salary"]

    submitted = data["total_compensation"]

    if submitted > max_salary:

        return {
            "benchmark_score": 30,
            "benchmark_issue":
            "Salary exceeds benchmark range"
        }

    elif submitted < min_salary:

        return {
            "benchmark_score": 40,
            "benchmark_issue":
            "Salary below benchmark range"
        }

    return {
        "benchmark_score": 100,
        "benchmark_issue":
        "Salary within benchmark range"
    }

# =========================================================
# ML ANOMALY DETECTION
# =========================================================

def ml_salary_anomaly(base_salary, tc, exp):

    global SEEN_COMPENSATIONS

    current = [base_salary, tc, exp]

    SEEN_COMPENSATIONS.append(current)

    if len(SEEN_COMPENSATIONS) < 10:
        return False

    isolation_model.fit(SEEN_COMPENSATIONS)

    prediction = isolation_model.predict([current])

    return prediction[0] == -1

# =========================================================
# MAIN AI VALIDATION ENGINE
# =========================================================

def analyze_submission(data):

    score = 100

    issues = []

    risk_score = 0

    company = normalize_text(data.get("company"))

    email = normalize_text(data.get("company_email"))

    location = normalize_text(data.get("location"))

    level = normalize_text(data.get("level_designation"))

    achievement = normalize_text(data.get("achievement"))

    session_id = normalize_text(data.get("session_id"))

    base_salary = float(data.get("base_salary", 0))

    total_comp = float(data.get("total_compensation", 0))

    experience = float(data.get("years_of_experience", 0))

    # =====================================================
    # EMAIL VALIDATION
    # =====================================================

    valid_email, normalized_email = validate_email_address(email)

    if not valid_email:

        score -= 40

        issues.append("Invalid email format")

    # =====================================================
    # DISPOSABLE EMAIL
    # =====================================================

    if check_disposable_email(email):

        score -= 40

        issues.append("Disposable email detected")

    # =====================================================
    # DUPLICATE SESSION
    # =====================================================

    if detect_duplicate_session(session_id):

        score -= 50

        issues.append("Duplicate session detected")

    # =====================================================
    # SPAM DETECTION
    # =====================================================

    if detect_email_spam(email):

        score -= 35

        issues.append("Spam behavior detected")

    # =====================================================
    # TOTAL COMP CHECK
    # =====================================================

    expected_tc = (
        base_salary
        + float(data.get("bonus", 0))
        + float(data.get("stock_compensation", 0))
    )

    if abs(expected_tc - total_comp) > 1000:

        score -= 30

        issues.append(
            "Total compensation mismatch"
        )

    # =====================================================
    # EXPERIENCE CHECK
    # =====================================================

    if experience > 20 and level == "l1":

        score -= 25

        issues.append(
            "Experience-level mismatch"
        )

    # =====================================================
    # INTERN SALARY CHECK
    # =====================================================

    if "intern" in data.get("role", "").lower():

        if base_salary > 200000:

            score -= 50

            issues.append(
                "Unrealistic intern salary"
            )

    # =====================================================
    # NLP FRAUD DETECTION
    # =====================================================

    suspicious = detect_suspicious_keywords(
        achievement
    )

    if suspicious:

        score -= 20

        issues.append(
            f"Suspicious keywords: {suspicious}"
        )

    # =====================================================
    # CLAIM SIMILARITY
    # =====================================================

    repeated, similarity = detect_repeated_claims(
        achievement
    )

    if repeated:

        score -= 35

        issues.append(
            f"Repeated claim similarity: {round(similarity,2)}"
        )

    # =====================================================
    # BENCHMARK VALIDATION
    # =====================================================

    benchmark_result = benchmark_validation(data)

    score -= (100 - benchmark_result["benchmark_score"]) * 0.3

    issues.append(
        benchmark_result["benchmark_issue"]
    )

    # =====================================================
    # ML SALARY PREDICTION
    # =====================================================

    predicted_salary = predict_salary(data)

    deviation = abs(
        total_comp - predicted_salary
    ) / predicted_salary

    if deviation > 0.50:

        score -= 40

        issues.append(
            "Salary deviates significantly from AI prediction"
        )

    elif deviation > 0.25:

        score -= 20

        issues.append(
            "Salary moderately deviates from AI prediction"
        )

    # =====================================================
    # ML ANOMALY DETECTION
    # =====================================================

    anomaly = ml_salary_anomaly(
        base_salary,
        total_comp,
        experience
    )

    if anomaly:

        score -= 35

        issues.append(
            "ML anomaly detection triggered"
        )

    # =====================================================
    # SCORE NORMALIZATION
    # =====================================================

    score = max(score, 0)

    # =====================================================
    # CONFIDENCE LABEL
    # =====================================================

    if score >= 85:

        confidence = "High"

        label = "VERIFIED"

    elif score >= 60:

        confidence = "Medium"

        label = "REVIEW_REQUIRED"

    else:

        confidence = "Low"

        label = "HIGH_RISK"

    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {

        "validation_label": label,

        "quality_score": round(score, 2),

        "confidence": confidence,

        "issues": issues,

        "flagged": score < 60,

        "ai_validation": {

            "predicted_salary":
                round(predicted_salary, 2),

            "expected_range": {

                "min":
                    round(predicted_salary * 0.85, 2),

                "max":
                    round(predicted_salary * 1.15, 2)
            },

            "salary_deviation":
                round(deviation * 100, 2),

            "ml_anomaly_detection":
                anomaly,

            "spam_probability":
                round(
                    min(SEEN_EMAILS[email] / 10, 1.0),
                    2
                )
        }
    }