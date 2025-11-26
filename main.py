import streamlit as st
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configure page
st.set_page_config(page_title="Class 11/12 Quiz Master", page_icon="🎓", layout="wide")

def init_session_state():
    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = None
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'checked_answers' not in st.session_state:
        st.session_state.checked_answers = {}

def get_available_models(api_key):
    try:
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append(m.name)
        return models
    except Exception as e:
        return []

def generate_quiz(api_key, model_name, subject, topic, difficulty, num_questions, explanation_depth):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        Act as an expert {subject} teacher for Class 11 and 12 students.
        Create a quiz about "{topic}" with {num_questions} conceptual and numerical questions.
        Difficulty level: {difficulty}.
        Explanation Depth: {explanation_depth} (Provide explanations accordingly).
        
        For EACH question, you MUST provide Python code using matplotlib, seaborn, or pandas to visualize the concept or the solution (e.g., a graph of the function, a diagram, or a data plot).
        The python code should be self-contained, use 'fig, ax = plt.subplots()' (or sns functions that return an ax/fig) and NOT call plt.show(). It should just create the plot on 'ax' or 'fig'.
        
        Return the output strictly as a valid JSON array of objects.
        Each object must have the following keys:
        - "question": The question text (string)
        - "options": A list of 4 possible answers (list of strings)
        - "answer": The correct answer text (string, must be exactly one of the options)
        - "explanation": A detailed explanation suitable for a student (string)
        - "visualization_code": Python code string to generate a matplotlib figure named 'fig'. Do not include markdown fences.
        
        Example of visualization_code:
        "import numpy as np\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns\\nfig, ax = plt.subplots()\\nx = np.linspace(0, 10, 100)\\ny = np.sin(x)\\nsns.lineplot(x=x, y=y, ax=ax)\\nax.set_title('Sine Wave')"
        
        Do not include any markdown formatting (like ```json), just the raw JSON string.
        """
        
        with st.spinner(f"Generating {num_questions} questions... This might take a minute."):
            response = model.generate_content(prompt)
            text = response.text.strip()
            
            # Clean up markdown
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            
            return json.loads(text)
            
    except ResourceExhausted:
        st.error(f"⚠️ Quota Exceeded for model '{model_name}'. Please select a different model (like gemini-1.5-flash) from the sidebar or wait a minute.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def execute_viz_code(code_str):
    try:
        # Create a local scope for execution
        local_scope = {}
        # Execute the code
        exec(code_str, globals(), local_scope)
        # Retrieve the figure
        if 'fig' in local_scope:
            return local_scope['fig']
        return None
    except Exception as e:
        st.warning(f"Could not render visualization: {e}")
        return None

def main():
    init_session_state()
    
    st.title("🎓 Class 11/12 Quiz Master")
    st.markdown("""
<style>
.big-font {
    font-size: 400px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

    st.markdown('<p class="big-font">Creator Dwarika Kohar</p>', unsafe_allow_html=True)
    st.caption("Physics | Chemistry | Mathematics | Biology")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        api_key_input = st.text_input("Gemini API Key", type="password")
        if st.button("Save Key"):
            st.session_state.api_key = api_key_input
            st.success("Key saved!")
            st.rerun()
            
        if st.session_state.api_key:
            models = get_available_models(st.session_state.api_key)
            if models:
                model_name = st.selectbox("Select Model", models, index=0)
            else:
                st.error("No suitable models found or Invalid Key.")
                model_name = "models/gemini-1.5-flash" # Fallback
        else:
            model_name = "models/gemini-1.5-flash"

        st.divider()
        st.markdown("### Quiz Configuration")
        subject = st.selectbox("Subject", ["Physics", "Chemistry", "Mathematics", "Biology"])
        difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"])
        
        num_questions = st.slider("Number of Questions", min_value=1, max_value=25, value=5)
        explanation_depth = st.select_slider("Explanation Depth", options=["Brief", "Standard", "Detailed"], value="Standard")

    if not st.session_state.api_key:
        st.info("Please enter your Gemini API Key in the sidebar to start.")
        st.markdown("👉 **Don't have an API key?** [Get one here from Google AI Studio](https://aistudio.google.com/app/apikey)")
        return

    # Quiz Generation
    if st.session_state.quiz_data is None:
        with st.form("gen_form"):
            topic = st.text_input("Enter Topic (e.g., Thermodynamics, Calculus, Genetics)")
            submitted = st.form_submit_button("Generate Quiz")
            
            if submitted and topic:
                data = generate_quiz(st.session_state.api_key, model_name, subject, topic, difficulty, num_questions, explanation_depth)
                if data:
                    st.session_state.quiz_data = data
                    st.session_state.user_answers = {}
                    st.session_state.score = 0
                    st.session_state.submitted = False
                    st.session_state.current_question = 0
                    st.session_state.checked_answers = {}
                    st.rerun()

    # Quiz Display
    else:
        q_idx = st.session_state.current_question
        q_data = st.session_state.quiz_data[q_idx]
        total_q = len(st.session_state.quiz_data)
        
        st.subheader(f"Question {q_idx + 1} of {total_q}")
        st.progress((q_idx + 1) / total_q)
        
        st.markdown(f"### {q_data['question']}")
        
        # Layout: Options on left, Visualization on right (if available)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Use a form for the current question to handle selection state better
            # or just use radio with key based on index
            
            # Check if this question has been answered/checked
            is_checked = st.session_state.checked_answers.get(q_idx, False)
            
            # Get previous selection if exists
            prev_selection = st.session_state.user_answers.get(q_idx, None)
            
            # We need to find the index of the previous selection in the options list
            try:
                idx = q_data['options'].index(prev_selection) if prev_selection else None
            except ValueError:
                idx = None

            selection = st.radio(
                "Select Answer:",
                q_data['options'],
                key=f"q_{q_idx}",
                index=idx,
                disabled=is_checked
            )
            
            if selection:
                st.session_state.user_answers[q_idx] = selection

            if not is_checked:
                if st.button("Check Answer"):
                    if selection:
                        st.session_state.checked_answers[q_idx] = True
                        if selection == q_data['answer']:
                            st.session_state.score += 1
                        st.rerun()
                    else:
                        st.warning("Please select an answer first.")
            else:
                # Show feedback
                if st.session_state.user_answers.get(q_idx) == q_data['answer']:
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Incorrect. Answer: {q_data['answer']}")
                
                with st.expander("Explanation", expanded=True):
                    st.info(q_data['explanation'])

        with col2:
            if q_data.get('visualization_code'):
                with st.expander("Show Concept Visualizer", expanded=True):
                    fig = execute_viz_code(q_data['visualization_code'])
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)

        st.divider()
        
        # Navigation Buttons
        c1, c2, c3 = st.columns([1, 2, 1])
        
        with c1:
            if q_idx > 0:
                if st.button("⬅️ Previous"):
                    st.session_state.current_question -= 1
                    st.rerun()
        
        with c3:
            if q_idx < total_q - 1:
                if st.button("Next ➡️"):
                    st.session_state.current_question += 1
                    st.rerun()
            else:
                if st.button("Finish Quiz 🏆"):
                    st.session_state.submitted = True
                    st.rerun()

        if st.session_state.submitted:
            st.balloons()
            st.success(f"Quiz Completed! Final Score: {st.session_state.score} / {total_q}")
            if st.button("Start New Quiz"):
                st.session_state.quiz_data = None
                st.rerun()

if __name__ == "__main__":
    main()
