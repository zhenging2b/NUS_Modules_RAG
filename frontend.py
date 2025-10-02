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

        # Extract last assistant message - FIXED
        # The result has structure: {"generate": {"messages": [...]}}
        if "generate" in result["result"]:
            messages = result["result"]["generate"]["messages"]
        elif "query_or_respond" in result["result"]:
            messages = result["result"]["query_or_respond"]["messages"]

        if messages:
            # Find the last AI message
            for msg in reversed(messages):
                if hasattr(msg, 'content'):
                    answer = msg.content
                    break
                elif isinstance(msg, dict) and 'content' in msg:
                    answer = msg['content']
                    break
            else:
                answer = "No response."
        else:
            answer = f"Unexpected response format. Result: {result['result']}"

    # Display
    st.write("### Answer:")
    st.write(answer)
