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
        st.session_state.score = None
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""

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

def generate_quiz(api_key, model_name, subject, topic, difficulty):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        Act as an expert {subject} teacher for Class 11 and 12 students.
        Create a quiz about "{topic}" with 5 conceptual and numerical questions.
        Difficulty level: {difficulty}.
        
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
        
        with st.spinner("Generating quiz and visualizations..."):
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

    if not st.session_state.api_key:
        st.info("Please enter your Gemini API Key in the sidebar to start.")
        return

    # Quiz Generation
    if st.session_state.quiz_data is None:
        with st.form("gen_form"):
            topic = st.text_input("Enter Topic (e.g., Thermodynamics, Calculus, Genetics)")
            submitted = st.form_submit_button("Generate Quiz")
            
            if submitted and topic:
                data = generate_quiz(st.session_state.api_key, model_name, subject, topic, difficulty)
                if data:
                    st.session_state.quiz_data = data
                    st.session_state.user_answers = {}
                    st.session_state.score = None
                    st.session_state.submitted = False
                    st.rerun()

    # Quiz Display
    else:
        st.subheader(f"Topic: {st.session_state.quiz_data[0].get('topic', 'Quiz')}")
        
        with st.form("quiz_form"):
            for i, q in enumerate(st.session_state.quiz_data):
                st.markdown(f"### Q{i+1}: {q['question']}")
                
                # Layout: Options on left, Visualization on right (if available)
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    sel = st.radio(
                        "Select Answer:",
                        q['options'],
                        key=f"q_{i}",
                        index=None,
                        disabled=st.session_state.submitted
                    )
                    if sel:
                        st.session_state.user_answers[i] = sel
                
                with col2:
                    if q.get('visualization_code'):
                        with st.expander("Show Concept Visualizer", expanded=True):
                            fig = execute_viz_code(q['visualization_code'])
                            if fig:
                                st.pyplot(fig)
                                plt.close(fig) # Clean up memory

                if st.session_state.submitted:
                    if st.session_state.user_answers.get(i) == q['answer']:
                        st.success("✅ Correct")
                    else:
                        st.error(f"❌ Incorrect. Answer: {q['answer']}")
                    st.info(f"**Explanation:** {q['explanation']}")
                
                st.divider()
            
            if not st.session_state.submitted:
                if st.form_submit_button("Submit Quiz"):
                    st.session_state.submitted = True
                    score = sum(1 for i, q in enumerate(st.session_state.quiz_data) 
                              if st.session_state.user_answers.get(i) == q['answer'])
                    st.session_state.score = score
                    st.rerun()
            else:
                st.write(f"### Final Score: {st.session_state.score} / {len(st.session_state.quiz_data)}")
                if st.form_submit_button("Start New Quiz"):
                    st.session_state.quiz_data = None
                    st.rerun()

if __name__ == "__main__":
    main()