"""Streamlit UI for ScholarMind."""
from __future__ import annotations
import streamlit as st
import requests
import plotly.express as px
import pandas as pd

API = st.secrets.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ScholarMind", page_icon="🧠", layout="wide")
st.title("🧠📚 ScholarMind — Multi-Agent Literature Reviewer")

with st.sidebar:
    st.markdown("### Knowledge Graph")
    if st.button("Refresh KG summary"):
        try:
            s = requests.get(f"{API}/kg/summary", timeout=10).json()
            st.json(s)
        except Exception as e:
            st.error(e)

    concept = st.text_input("Lookup concept", "graph neural network")
    if st.button("Find papers") and concept:
        try:
            data = requests.get(f"{API}/kg/concept/{concept}", timeout=15).json()
            for p in data.get("papers", []):
                st.markdown(f"- [{p['title']}]({p.get('url','#')})")
        except Exception as e:
            st.error(e)

q = st.text_area("Research question",
                 "Survey self-supervised learning methods for graph neural networks.",
                 height=80)

if st.button("🔍 Run multi-agent review", type="primary") and q:
    with st.spinner("Agents thinking…"):
        try:
            resp = requests.post(f"{API}/query", json={"question": q}, timeout=300).json()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Sources used", resp.get("n_sources", 0))
    col2.metric("Citations verified", resp.get("n_supported_citations", 0))
    col3.metric("Citations total", resp.get("n_total_citations", 0))

    if resp.get("sources"):
        st.write("Sources:")
        for s in resp.get("sources", []):
            st.write("-", s)

    sub_questions = resp.get("sub_questions", [])
    if sub_questions:
        st.subheader("Sub-questions")
        for i, sq in enumerate(sub_questions, 1):
            st.markdown(f"{i}. {sq}")

    st.subheader("📄 Survey draft")
    st.markdown(resp.get("answer", "_No answer returned from API._"))

    n_total = resp.get("n_total_citations", 0)
    n_supported = resp.get("n_supported_citations", 0)
    if n_total:
        df = pd.DataFrame({
            "type": ["supported", "unsupported"],
            "count": [n_supported, n_total - n_supported],
        })
        st.plotly_chart(px.pie(df, values="count", names="type",
                               title="Citation grounding"), use_container_width=True)
