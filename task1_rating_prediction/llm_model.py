from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from typing import Literal, Optional
from dotenv import load_dotenv
import os

# load all environment variables
load_dotenv()

'''
The following class is a wrapper class for the LLMs.
Currently, it supports two providers:---- [Groq ---- and ---- OpenRouter] ----.
------------------------------------Main Model Class------------------------------------
'''

class ChatLLM:
    def __init__(self, provider: Literal['groq', 'openrouter'], model_name: Optional[str] = None):
        self.provider = provider
        self.model_name = model_name
        
        # select llm based on user's preference
        self.model = self._get_model()

    def _get_model(self):
        if self.provider == 'groq':
            return self._create_groq_model()
        elif self.provider == 'openrouter':
            return self._create_openrouter_model()
        else:
            raise ValueError(f"Provider '{self.provider}' is not supported.")

    def _create_groq_model(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        model_name = self.model_name
        
        return ChatGroq(
            api_key=api_key,
            model=model_name
        )

    def _create_openrouter_model(self):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

        model_name = self.model_name or 'amazon/nova-2-lite-v1:free'
        
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,  
            model=model_name,
        )


if __name__ == '__main__':
    try:
        # Example usage
        print("Initializing Groq model...")
        model = ChatLLM(provider='groq')
        print(f"Model created: {model.model}")
        
        # Uncomment to test actual invocation if keys are valid
        res = model.model.invoke('Hey what is the weather like today?')
        print(res)
        
    except Exception as e:
        print(f"Error: {e}")