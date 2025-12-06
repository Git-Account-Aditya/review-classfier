from langchain_core.prompts import PromptTemplate
import textwrap

def get_prompt_v3():
    SYSTEM_PROMPT = """
    You are a sophisticated AI designed to predict product ratings based on customer reviews.

    Your goal is to accurately assign a star rating (1-5) by following a logical reasoning process.

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

    Few-Shot Examples:

    Example 1:
    Review Text: "Absolutely love this phone! The battery life is stellar and the camera takes professional-quality photos. Best purchase I've made all year."
    Reactions: 'cool': 5, 'useful': 45, 'funny': 0
    Output:
    {{
        "predicted_stars": 5,
        "explanation": "The reviewer uses highly positive language ('Absolutely love', 'stellar', 'Best purchase'). Specific features like battery life and camera are praised. High 'useful' count supports it is a helpful review."
    }}

    Example 2:
    Review Text: "This product is a complete waste of money. It broke after two days of use. Customer service was unhelpful and rude. Do not buy!"
    Reactions: 'cool': 0, 'useful': 15, 'funny': 2
    Output:
    {{
        "predicted_stars": 1,
        "explanation": "The review expresses strong dissatisfaction ('complete waste of money', 'Do not buy'). It cites product failure and poor customer service. The tone is angry."
    }}

    Example 3:
    Review Text: "The sound quality is decent for the price, but the earcups are uncomfortable after an hour. It's okay for short commutes but not for long listening sessions."
    Reactions: 'cool': 1, 'useful': 5, 'funny': 0
    Output:
    {{
        "predicted_stars": 3,
        "explanation": "The reviewer recognizes value ('decent for the price', 'okay for short commutes') but highlights a significant flaw ('uncomfortable'). The sentiment is mixed, balancing pros and cons."
    }}

    Now analyze the following case:

    Input Data:
    Review Text: {review_text}
    Reactions: {reactions}

    """
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=['review_text', 'reactions'],
    )
    return prompt
