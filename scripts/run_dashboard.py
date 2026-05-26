import os
import sys


if __name__ == "__main__":
    # os.system() runs a shell command
    # streamlit run = the command to start a Streamlit app
    # --server.port 8501 = which port to serve on
    # --server.headless false = open browser automatically
    dashboard_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "app", "dashboard", "app.py"
    )
    os.system(f"streamlit run {dashboard_path} --server.port 8501")