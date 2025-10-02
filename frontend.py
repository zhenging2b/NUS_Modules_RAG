import streamlit as st
import requests

st.title("NUS Module Helper")

# Keep thread_id across v2 interactions
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# Dropdown to select version
version = st.selectbox("Choose version:", ["v1", "v2"])

user_input = st.text_input("Ask about modules:")

if st.button("Submit"):
    if version == "v1":
        # Call v1 endpoint
        payload = {"question": user_input}
        response = requests.post("http://127.0.0.1:8000/ask-v1", json=payload)
        result = response.json()
        answer = result["result"]["generate"]["answer"]

    else:  # v2
        payload = {
            "question": user_input,
            "thread_id": st.session_state.thread_id
        }
        response = requests.post("http://127.0.0.1:8000/ask-v2", json=payload)
        result = response.json()

        # Save thread_id for persistence
        st.session_state.thread_id = result["thread_id"]

        # Extract last assistant message
        messages = result["result"].get("messages", [])
        if messages:
            answer = messages[-1]["content"]
        else:
            answer = "No response."

    # Display
    st.write("### Answer:")
    st.write(answer)
