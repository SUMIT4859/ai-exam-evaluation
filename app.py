import streamlit as st
import cv2
import numpy as np
from evaluate import evaluate_exam, similarity_score, extract_text, split_answers
from answer_key import correct_answers


# ---------------- Page Setup ----------------
st.set_page_config(
    page_title="Exam Evaluation System",
    page_icon="📚",
    layout="centered"
)

st.title("📚 AI-Based Exam Evaluation System")

mode = st.radio(
    "Select Evaluation Mode",
    ["Full Exam (20 Marks)", "Single Question (5 Marks)"]
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload Answer Sheet Image",
    type=["jpg", "png", "jpeg"]
)

# ---------------- When Image Uploaded ----------------
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Answer", width=500)
    st.markdown("---")

    # ================= FULL EXAM MODE =================
    if mode == "Full Exam (20 Marks)":

        results, total_marks = evaluate_exam(image)

        if results is None:
            st.error("❌ Unable to detect proper answers. Please upload a clear image.")
        else:
            st.subheader("📊 Evaluation Result")

            for q, data in results.items():
                st.write(
                    f"**{q} → {data['marks']}/5**  |  Similarity: {data['similarity']:.2f}%"
                )

            st.markdown("---")

            percentage = (total_marks / 20) * 100

            # -------- Grade Logic --------
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

            st.markdown(f"### 🎯 Total Marks: {total_marks}/20")
            st.progress(int(percentage))
            st.markdown(f"### 📈 Percentage: {percentage:.2f}%")

            st.markdown(
                f"<h3 style='color:{color};'>🏅 Result: {grade}</h3>",
                unsafe_allow_html=True
            )

    # ================= SINGLE QUESTION MODE =================
    else:

        question = st.selectbox(
            "Select Question",
            list(correct_answers.keys())
        )

        full_text = extract_text(image)

        # Step 1: OCR Validation
        if len(full_text.strip()) < 15:
            st.error("❌ Unable to read text from image. Please upload clearer image.")

        else:
            answers = split_answers(full_text)

            # Step 2: Check if any question label detected
            if all(len(ans.strip()) == 0 for ans in answers.values()):
                st.error("❌ No valid question labels (Q1, Q2, Q3, Q4) detected.")

            else:
                student_answer = answers.get(question, "").strip()
                correct_text = correct_answers[question]

                # Step 3: Check selected question answer
                if len(student_answer) < 5:
                    st.warning("⚠ Selected question answer not detected properly.")
                    similarity = 0
                    marks = 0
                else:
                    similarity = similarity_score(student_answer, correct_text)

                    # Same marking logic as evaluate.py
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

                st.subheader("📊 Evaluation Result")
                st.write(f"**{question} → {marks}/5**")
                st.progress(int(similarity))
                st.write(f"Similarity: {similarity:.2f}%")

                # For Demo Purpose
                with st.expander("🔍 View Detected Answer Text"):
                    st.write(student_answer if student_answer else "No answer detected.")