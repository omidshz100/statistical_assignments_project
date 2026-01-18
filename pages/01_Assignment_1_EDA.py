import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Assignment 1 - EDA", page_icon="📈", layout="wide")
st.title("📈 Assignment 1: Exploratory Data Analysis")

# Import and run assignment1 code
from assignment1 import functions_from_1  # Adjust based on your code structure

# Display your EDA content here