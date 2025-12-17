import streamlit as st
import os
import subprocess
import sys
from datetime import datetime

st.set_page_config(page_title="Neural Style Transfer", layout="centered")
st.title("Neural Style Transfer")

# ================= FILE UPLOAD =================

content = st.file_uploader("Content Image", type=["png", "jpg", "jpeg"])
style = st.file_uploader("Style Image", type=["png", "jpg", "jpeg"])

# ================= PARAMETER CONTROLS =================

height = st.slider("Height", 128, 1024, 400)
content_weight = st.number_input("Content Weight", value=100000.0)
style_weight = st.number_input("Style Weight", value=30000.0)
tv_weight = st.number_input("TV Weight", value=1.0)
num_iter = st.slider("Iterations", 100, 2000, 1000)

# ================= RUN =================

if st.button("Run NST"):
    if not content or not style:
        st.error("Please upload both images.")
        st.stop()

    # ---------- Create run directory ----------
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # ---------- Save images ----------
    with open(os.path.join(run_dir, "content_image.png"), "wb") as f:
        f.write(content.read())
    with open(os.path.join(run_dir, "style_image.png"), "wb") as f:
        f.write(style.read())

    st.success(f"Running NST in: {run_dir}")
    st.info("Check PowerShell console for iteration losses.")

    # ---------- IMPORTANT PART ----------
    # NO stdout/stderr redirection
    # -u = unbuffered → prints every iteration immediately
    subprocess.Popen(
        [
            sys.executable, "-u", "NST.py",
            "--run_dir", run_dir,
            "--height", str(height),
            "--content_weight", str(content_weight),
            "--style_weight", str(style_weight),
            "--tv_weight", str(tv_weight),
            "--num_of_iterations", str(num_iter),
        ],
        cwd=os.getcwd()
    )
