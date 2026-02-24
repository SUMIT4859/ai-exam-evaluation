# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------

import streamlit as st                     # For building web interface
import cv2                                # For image processing
import numpy as np                        # For handling image data
from evaluate import evaluate_exam, similarity_score, extract_text, split_answers
from answer_key import correct_answers     # Import correct answers


# ------------------------------------------------------------
# 1️⃣ PAGE SETUP
# ------------------------------------------------------------

# Configure page title and layout
st.set_page_config(
    page_title="Exam Evaluation System",
    page_icon="📚",
    layout="centered"
)

# Display main title
st.title("📚 Exam Evaluation System")


# ------------------------------------------------------------
# 2️⃣ SELECT EVALUATION MODE
# ------------------------------------------------------------

# User can choose between full exam or single question evaluation
mode = st.radio(
    "Select Evaluation Mode",
    ["Full Exam (20 Marks)", "Single Question (5 Marks)"]
)

st.markdown("---")


# ------------------------------------------------------------
# 3️⃣ IMAGE UPLOAD
# ------------------------------------------------------------

# Allow user to upload answer sheet image
uploaded_file = st.file_uploader(
    "Upload Answer Sheet Image",
    type=["jpg", "png", "jpeg"]
)


# ------------------------------------------------------------
# 4️⃣ WHEN IMAGE IS UPLOADED
# ------------------------------------------------------------

if uploaded_file is not None:

    # Convert uploaded file into image format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Show uploaded image on screen
    st.image(image, caption="Uploaded Answer", width=500)
    st.markdown("---")


    # ============================================================
    # 5️⃣ FULL EXAM MODE (20 MARKS)
    # ============================================================

    if mode == "Full Exam (20 Marks)":

        # Call main evaluation function
        results, total_marks = evaluate_exam(image)

        # If image is invalid
        if results is None:
            st.error("❌ Unable to detect proper answers. Please upload a clear image.")

        else:
            st.subheader("📊 Evaluation Result")

            # Show result for each question
            for q, data in results.items():
                st.write(
                    f"**{q} → {data['marks']}/5**  |  Similarity: {data['similarity']:.2f}%"
                )

            st.markdown("---")

            # Calculate percentage
            percentage = (total_marks / 20) * 100

            # ----------------------------------------------------
            # Grade Logic
            # ----------------------------------------------------

            if percentage < 35:
                grade = "Fail"
                color = "red"
            elif percentage < 55:
                grade = "Pass"
                color = "orange"
            elif percentage < 65:
                grade = "Second Class"
                color = "blue"
            elif percentage < 75:
                grade = "First Class"
                color = "green"
            else:
                grade = "First Class with Distinction"
                color = "darkgreen"

            # Display total marks and grade
            st.markdown(f"### 🎯 Total Marks: {total_marks}/20")
            st.progress(int(percentage))
            st.markdown(f"### 📈 Percentage: {percentage:.2f}%")

            st.markdown(
                f"<h3 style='color:{color};'>🏅 Result: {grade}</h3>",
                unsafe_allow_html=True
            )


    # ============================================================
    # 6️⃣ SINGLE QUESTION MODE (5 MARKS)
    # ============================================================

    else:

        # Let user select which question to evaluate
        question = st.selectbox(
            "Select Question",
            list(correct_answers.keys())
        )

        # Extract full text using OCR
        full_text = extract_text(image)

        # Step 1: Check if OCR detected text
        if len(full_text.strip()) < 15:
            st.error("❌ Unable to read text from image. Please upload clearer image.")

        else:
            # Split answers based on Q1, Q2, Q3, Q4
            answers = split_answers(full_text)

            # Step 2: Check if question labels found
            if all(len(ans.strip()) == 0 for ans in answers.values()):
                st.error("❌ No valid question labels detected.")

            else:
                # Get selected question answer
                student_answer = answers.get(question, "").strip()
                correct_text = correct_answers[question]

                # If answer not detected properly
                if len(student_answer) < 5:
                    st.warning("⚠ Selected question answer not detected properly.")
                    similarity = 0
                    marks = 0
                else:
                    # Calculate similarity
                    similarity = similarity_score(student_answer, correct_text)

                    # Assign marks based on similarity
                    if similarity >= 75:
                        marks = 5
                    elif similarity >= 55:
                        marks = 4
                    elif similarity >= 45:
                        marks = 3
                    elif similarity >= 35:
                        marks = 2
                    else:
                        marks = 0

                # Show result
                st.subheader("📊 Evaluation Result")
                st.write(f"**{question} → {marks}/5**")
                st.progress(int(similarity))
                st.write(f"Similarity: {similarity:.2f}%")

                # Show detected answer text (for demo)
                with st.expander("🔍 View Detected Answer Text"):
                    st.write(student_answer if student_answer else "No answer detected.")
