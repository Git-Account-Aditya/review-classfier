from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from structured_output import StructuredOutput

def get_prompt_v3():
    SYSTEM_PROMPT = """
    You are a sophisticated AI designed to predict product ratings based on customer reviews.

    Your goal is to accurately assign a star rating (1-5) by following a logical reasoning process.

    Input Data:
    Review Text: {review_text}
    Reactions: {reactions}

    Detailed Analysis Step-by-Step:
    1. **Sentiment Identification**: Is the reviewer happy, angry, disappointed, or neutral?
    2. **Key Feature Extraction**: Identify specific praises (e.g., "amazing battery") or complaints (e.g., "arrived broken").
    3. **Intensity Evaluation**: specific words like "terrible" vs "not great" indicate different star levels (1 vs 2/3).
    4. **Contextual Clues**: Use reactions to gauge community consensus if applicable.
    5. **Rating Assignment**:
       - 1 Star: Severe issues, highly negative.
       - 2 Stars: Mostly negative, major flaws.
       - 3 Stars: Mediocre, mixed feelings.
       - 4 Stars: Good but not perfect.
       - 5 Stars: Excellent, highly recommended.

    Output Instructions:
    - First, formulate your explanation based on the steps above.
    - Then, determine the final `predicted_stars`.

    {instructions}
    """

    parser = PydanticOutputParser(pydantic_object=StructuredOutput)

    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=['review_text', 'reactions'],
        partial_variables={'instructions': parser.get_format_instructions()}
    )

    return prompt, parser
