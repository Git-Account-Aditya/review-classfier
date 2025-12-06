from pydantic import BaseModel, Field
from typing import List, Tuple, Literal, Optional
from dotenv import load_dotenv
import os

class StructuredOutput(BaseModel):
    predicted_stars: Literal[1, 2, 3, 4, 5] = Field(description="The predicted number of stars given by user to the product.")
    explanation: str = Field(description="The explanation for the predicted number of stars.")