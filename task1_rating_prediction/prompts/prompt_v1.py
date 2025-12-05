from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from typing import List, Tuple, Literal, Optional
from dotenv import load_dotenv
import os

# load all environment variables
load_dotenv()


class StructuredOutput(BaseModel):
    predicted_stars: Literal[1, 2, 3, 4, 5] = Field(description="The predicted number of stars given by user to the product.")
    explanation: str = Field(description="The explanation for the predicted number of stars.")

def get_prompt_v1():
    SYSTEM_PROMPT = """
    You are a helpful assistant that can predict the number of stars given by user to the product based on the review text.

    Here is the review text:
    {review_text}

    Use the structured output format to return your prediction.
    {instructions}
    """

    parser = PydanticOutputParser(pydantic_object=StructuredOutput)

    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=['review_text'],
        partial_variables={'instructions': parser.get_format_instructions()}
    )

    return prompt, parser
    
    