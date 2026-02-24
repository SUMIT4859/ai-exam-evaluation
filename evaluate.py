# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------

import cv2                          # For image processing
import numpy as np                  # For numerical operations
import pytesseract                  # For OCR (Image → Text)
import re                           # For text cleaning & pattern matching
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text to numbers
from sklearn.metrics.pairwise import cosine_similarity       # Compare two texts
from answer_key import correct_answers  # Import correct answers dictionary


# ------------------------------------------------------------
# 1️⃣ TEXT CLEANING FUNCTION
# ------------------------------------------------------------
# This function prepares text for comparison.
# It removes unwanted characters and standardizes format.

def clean_text(text):

    text = text.lower()                          # Convert all letters to lowercase
    text = re.sub(r'\s+', ' ', text)             # Remove extra spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)      # Remove special characters

    return text.strip()                          # Remove space from start & end


# ------------------------------------------------------------
# 2️⃣ EXTRACT TEXT FROM IMAGE USING OCR
# ------------------------------------------------------------
# This function converts answer sheet image into text.

def extract_text(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

    # Apply threshold to improve text clarity
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    # Use Tesseract OCR to read text from image
    return pytesseract.image_to_string(gray)


# ------------------------------------------------------------
# 3️⃣ VALIDATE IF IMAGE HAS MEANINGFUL TEXT
# ------------------------------------------------------------
# This prevents random images (like selfies) from being evaluated.

def is_valid_text(full_text):

    cleaned = clean_text(full_text)

    # If text length is small, consider image invalid
    return len(cleaned) > 20


# ------------------------------------------------------------
# 4️⃣ SPLIT FULL TEXT INTO Q1, Q2, Q3, Q4
# ------------------------------------------------------------
# This function separates answers based on question numbers.

def split_answers(full_text):

    answers = {}                          # Store answers here
    full_text = full_text.lower()         # Convert text to lowercase

    question_keys = list(correct_answers.keys())  # ['Q1', 'Q2', 'Q3', 'Q4']

    for q in question_keys:

        # Create pattern to extract text after Q1 until next question
        pattern = (
            q.lower() +
            r'[:\-]?\s*(.+?)(?=' +
            '|'.join([k.lower() for k in question_keys]) +
            r'[:\-]?\s*|$)'
        )

        match = re.search(pattern, full_text, re.DOTALL)

        if match:
            answers[q] = match.group(1)  # Extract answer text
        else:
            answers[q] = ""              # If not found, keep empty

    return answers


# ------------------------------------------------------------
# 5️⃣ CALCULATE SIMILARITY BETWEEN TWO TEXTS
# ------------------------------------------------------------
# Uses TF-IDF + Cosine Similarity

def similarity_score(student_text, correct_text):

    # Clean both texts
    student_text = clean_text(student_text)
    correct_text = clean_text(correct_text)

    # If student answer is empty → 0%
    if len(student_text) == 0:
        return 0.0

    # Convert text into numerical vectors
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([student_text, correct_text])

    # Calculate cosine similarity between vectors
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])

    # Convert similarity score into percentage
    return float(score[0][0]) * 100


# ------------------------------------------------------------
# 6️⃣ MAIN FUNCTION – COMPLETE EXAM EVALUATION
# ------------------------------------------------------------
# This function combines everything:
# OCR → Validation → Split → Similarity → Marks

def evaluate_exam(image):

    # Step 1: Extract full text from image
    full_text = extract_text(image)

    # Step 2: Validate if image contains readable text
    if not is_valid_text(full_text):
        return None, 0

    # Step 3: Split answers by question number
    answers = split_answers(full_text)

    # Step 4: If no question labels found → invalid
    if all(len(ans.strip()) == 0 for ans in answers.values()):
        return None, 0

    results = {}       # Store per-question results
    total_marks = 0    # Store total marks

    # Step 5: Evaluate each question
    for q, correct_text in correct_answers.items():

        student_answer = answers[q].strip()

        # If student did not write answer
        if len(student_answer) < 5:
            similarity = 0
            marks = 0
        else:
            # Calculate similarity percentage
            similarity = similarity_score(student_answer, correct_text)

            # Assign marks based on similarity
            if similarity >= 75:
                marks = 5
            elif similarity >= 45:
                marks = 3
            elif similarity >= 35:
                marks = 2
            else:
                marks = 0

        # Add marks to total
        total_marks += marks

        # Store result for that question
        results[q] = {
            "similarity": similarity,
            "marks": marks
        }

    # Return final result
    return results, total_marks
