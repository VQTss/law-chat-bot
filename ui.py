import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

st.title('🦜🔗 Law App')

# Sidebar for OpenAI API Key input
openai_api_key = st.sidebar.text_input('OpenAI API Key', type="password")

# Initialize session state for message history if not already done
if "history" not in st.session_state:
    st.session_state.history = []

def generate_response(input_text):
    try:
        # Ensure the API key is valid
        if not openai_api_key or not openai_api_key.startswith('sk-'):
            st.warning('Invalid OpenAI API Key! Please check and try again.', icon='⚠')
            return
        
        # Initialize ChatOpenAI with the provided API key
        llm = ChatOpenAI(
            temperature=0.7,
            openai_api_key=openai_api_key,
            model="gpt-4o-mini"
        )
        
        # Generate response using the chat model
        response = llm([HumanMessage(content=input_text)])
        
        # Save the input and response to history
        st.session_state.history.append({"user": input_text, "bot": response.content})
        
        # Display the response
        st.info(response.content)
        
    except Exception as e:
        st.error(f"An error occurred while processing your request: {str(e)}")

# Form to input text and submit
with st.form('my_form'):
    text = st.text_area('Enter text:', 'Input the question text')
    submitted = st.form_submit_button('Submit')
    
    # Validate API key and handle submission
    if not openai_api_key:
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    elif submitted:
        generate_response(text)

# Display message history
if st.session_state.history:
    st.subheader("Conversation History")
    for idx, message_pair in enumerate(st.session_state.history, 1):
        st.markdown(f"**Message {idx}:**")
        st.markdown(f"- **User:** {message_pair['user']}")
        st.markdown(f"- **Bot:** {message_pair['bot']}")
