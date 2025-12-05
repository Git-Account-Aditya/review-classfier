from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from typing import List, Tuple, Literal, Optional
from dotenv import load_dotenv
import os