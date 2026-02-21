import cv2
import numpy as np
import pytesseract
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from answer_key import correct_answers

# 🔹 Set Tesseract Path (required for OCR)



# --------------------------------------------------
# 1️⃣ Clean Text Function
# --------------------------------------------------
# This removes special characters and extra spaces
# and converts text to lowercase for better comparison
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()


# --------------------------------------------------
# 2️⃣ Extract Text from Image using OCR
# --------------------------------------------------
def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve OCR clarity using thresholding
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    return pytesseract.image_to_string(gray)


# --------------------------------------------------
# 3️⃣ Validate Image Contains Meaningful Text
# --------------------------------------------------
def is_valid_text(full_text):
    cleaned = clean_text(full_text)
    return len(cleaned) > 20  # Minimum text length required


# --------------------------------------------------
# 4️⃣ Split Answers Based on Q1, Q2, Q3, Q4
# --------------------------------------------------
def split_answers(full_text):
    answers = {}
    full_text = full_text.lower()

    question_keys = list(correct_answers.keys())

    for q in question_keys:
        pattern = (
            q.lower() +
            r'[:\-]?\s*(.+?)(?=' +
            '|'.join([k.lower() for k in question_keys]) +
            r'[:\-]?\s*|$)'
        )

        match = re.search(pattern, full_text, re.DOTALL)

        if match:
            answers[q] = match.group(1)
        else:
            answers[q] = ""

    return answers


# --------------------------------------------------
# 5️⃣ Calculate Similarity using TF-IDF + Cosine Similarity
# --------------------------------------------------
def similarity_score(student_text, correct_text):

    student_text = clean_text(student_text)
    correct_text = clean_text(correct_text)

    if len(student_text) == 0:
        return 0.0

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([student_text, correct_text])

    score = cosine_similarity(tfidf[0:1], tfidf[1:2])

    return float(score[0][0]) * 100


# --------------------------------------------------
# 6️⃣ Main Evaluation Function
# --------------------------------------------------
def evaluate_exam(image):

    # Step 1: Extract full text
    full_text = extract_text(image)

    # Step 2: Validate image
    if not is_valid_text(full_text):
        return None, 0

    # Step 3: Split answers
    answers = split_answers(full_text)

    # Step 4: Ensure at least one Q label exists
    if all(len(ans.strip()) == 0 for ans in answers.values()):
        return None, 0

    results = {}
    total_marks = 0

    # Step 5: Evaluate each question
    for q, correct_text in correct_answers.items():

        student_answer = answers[q].strip()

        if len(student_answer) < 5:
            similarity = 0
            marks = 0
        else:
            similarity = similarity_score(student_answer, correct_text)

            # 🔹 Marking Scheme
            if similarity >= 75:
                marks = 5
            elif similarity >= 45:
                marks = 3
            elif similarity >= 35:
                marks = 2
            else:
                marks = 0

        total_marks += marks

        results[q] = {
            "similarity": similarity,
            "marks": marks
        }

    return results, total_marks