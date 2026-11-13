import os
import streamlit as st
from google import genai
from google.genai import types

SERVICE_NAME = "ServiceMate"

SYSTEM_PROMPT = """
You are ServiceMate, a friendly AI assistant for a service business.
You help customers with service enquiries, booking requests and general support.

Rules:
- Reply in the same language the customer writes in (Hindi, English or Hinglish).
- Be polite, short and clear.
- Never invent prices, discounts, availability or policies. If you do not know,
  say so and ask the customer to contact the team.
- To take a booking request, collect: name, service needed, preferred date and
  time, and a contact number. Then summarise it and say the team will confirm.
- Do not give medical, legal or financial advice.
"""

st.set_page_config(page_title=SERVICE_NAME)
st.title(SERVICE_NAME)
st.caption("सर्विस पूछताछ और बुकिंग में मदद करने वाला AI असिस्टेंट")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

if "chat" not in st.session_state:
    st.session_state.chat = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    st.session_state.history = []

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

prompt = st.chat_input("अपना सवाल लिखिए")
if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        try:
            reply = st.session_state.chat.send_message(prompt).text
        except Exception:
            reply = "अभी जवाब नहीं आ पाया, कृपया थोड़ी देर बाद दोबारा कोशिश कीजिए।"
        st.write(reply)
    st.session_state.history.append(("assistant", reply))
