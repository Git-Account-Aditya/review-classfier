from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from structured_output import StructuredOutput

def get_prompt_v1():
    SYSTEM_PROMPT = """
    You are a helpful assistant that can predict the number of stars given by user.
    Here is the review with multiple reactions:
    {review_text}
    {reactions}

    Use the structured output format to return your prediction.
    {instructions}
    """

    parser = PydanticOutputParser(pydantic_object=StructuredOutput)

    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=['review_text', 'reactions'],
        partial_variables={'instructions': parser.get_format_instructions()}
    )
    return prompt, parser
    
    