import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai

import time 
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os 

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

st.set_page_config(
    page_title="Multimodal AI Agent",
    layout="wide",
    page_icon="📹"
)

st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
    }
    .stTextArea textarea {
        height: 100px;
    }
    .footer {
        position: fixed;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
        text-align: center;
        font-size: 14px;
        color: gray;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #1E88E5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>Video Insights Agent</div>", unsafe_allow_html=True)

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

multimodal_Agent = initialize_agent()

video_file = st.file_uploader(
    "Upload a video file", type=['mp4','mov','avi'], help="Upload a video for AI analysis"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content. The AI agent will analyze and gather additional context if needed.",
        help="Provide specific questions or insights you want from the video."
    )

    if st.button("🔍 Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research:
                        {user_query}
                        
                        Provide a detailed, user-friendly, and actionable response.
                        """
                    )
                    response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])

                st.subheader("Analysis Result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("Upload a video file to begin analysis.")

st.markdown(
    """
    <div class='footer'>
        <b>By Chandan Kumar</b><br>
        <a href='https://www.linkedin.com/in/chandan-kumar-438aa3193/' target='_blank'>LinkedIn</a> |
        <a href='https://github.com/modelpath-dev' target='_blank'>GitHub</a> |
        <a href='mailto:cml.codes@gmail.com'>Gmail</a> |
        <a href='https://cv-eojb.vercel.app/' target='_blank'>Portfolio</a>
    </div>
    """,
    unsafe_allow_html=True
)
