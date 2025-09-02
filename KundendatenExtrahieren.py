import sys
import os

import streamlit as st
from datetime import datetime
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

from openai import RateLimitError

from dotenv import load_dotenv
import requests
import time

load_dotenv(override=True)


def get_llm(json_output=False, structured_output=None) -> BaseChatOpenAI:
    '''
    Retrieves an instance of the LLM model.
    '''

    model = ChatOpenAI(
        api_key=os.getenv('LLM_HUB_API_KEY'),
        base_url=os.getenv('LLM_HUB_API_BASE'),
        model=os.environ.get('LLM_MODEL'),
        temperature=os.getenv('LLM_TEMPERATURE', 0.0),
        max_tokens=os.getenv('LLM_MAX_REPLY_TOKENS', 500),
        streaming=False,
    )

    return model

def call_model(messages) -> str:
    """Ruft das LLM-Modell auf und verarbeitet die Antwort.
    
    Args:
        messages: Liste von HumanMessage Objekten für den Modell-Input
        
    Returns:
        Bereinigter Text der Modell-Antwort
        
    Raises:
        RuntimeError: Wenn nach allen Versuchen keine erfolgreiche Antwort erhalten wurde
    """
    max_retries = 5
    base_wait_time = 4  # Sekunden

    model = get_llm()
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Modell-Aufruf (Versuch {attempt + 1}/{max_retries})")
            response = model.invoke(messages)
            response_text = response.content.strip()
            
            if not response_text:
                raise ValueError("Leere Antwort vom Modell erhalten")
                
            logger.debug(f"Erfolgreiche Antwort erhalten: {len(response_text)} Zeichen")
            return response_text
            
        except RateLimitError as e:
            wait_time = base_wait_time * (2 ** attempt)  # Exponentielles Backoff
            logger.warning(f"Rate Limit erreicht. Warte {wait_time}s. Fehler: {e}")
            time.sleep(wait_time)
            
        except Exception as e:
            logger.error(f"Fehler beim Modell-Aufruf: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = base_wait_time * (2 ** attempt)
                logger.info(f"Warte {wait_time}s vor erneutem Versuch")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Modell-Aufruf nach {max_retries} Versuchen fehlgeschlagen") from e
    
    raise RuntimeError(f"Modell-Aufruf nach {max_retries} Versuchen fehlgeschlagen")


def run_prompt(system_prompt, user_prompt):
    """
    Runs a prompt on an email text
    """
    
    prompt = [
            ('system', system_prompt),
            ('user', user_prompt)
        ]
    
    response = call_model(prompt)

    return response





# Streamlit app
# Adjust the width of the Streamlit app
st.html("""
    <style>
        .stMainBlockContainer {
            max-width:60rem;
        }
    </style>
    """
)
st.title("Prompt Workshop")

# Input textarea for system prompt
st.subheader("KI-Anweisungen (System Prompt)")
system_prompt = st.text_area("Hier wird die KI instruiert...", height=200)

# Input textarea for user prompt
st.subheader("User Prompt")
user_prompt = st.text_area("Hier wird die Benutzeranfrage eingegeben...", height=200)

# Input textarea for customer email
st.subheader("Customer Email")
email_text = st.text_area("Hier kommt die Kunden-E-Mail rein...", height=300)

user_prompt = user_prompt + "\n\nKunden E-Mail: " + email_text

# Submit button
if st.button("Prompt Ausführen"):
    with st.spinner("Processing..."):
        if email_text.strip():
            # Process the email using classify_email function
            try:
                
                result = run_prompt(system_prompt, user_prompt)
                # Display the output
                
                st.markdown(f'**KI Ausgabewerte:**')
                st.code(result, height=500)
                
            except Exception as e:
                st.error(f"An error occurred while processing: {e}")
        else:
            st.warning("Please enter some data before submitting.")
