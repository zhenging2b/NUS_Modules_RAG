import streamlit as st
import requests

st.title("NUS Module Helper")

user_input = st.text_input("Ask about modules:")

if st.button("Submit"):
    response = requests.post("http://127.0.0.1:8000/ask", json={"question": user_input})
    result = response.json()
    st.write("### Answer:")
    st.write(result["result"]["generate"]["answer"])