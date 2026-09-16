from dotenv import load_dotenv
load_dotenv()

from pprint import pprint
from typing import Any, Dict, List, TypedDict
from tavily import TavilyClient
from pathlib import Path
import base64

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

""" class ResponseOutput(TypedDict):
    recipes: dict[str, str]
    ingredients: List[str]  """

system_prompt = f"""
    You are an expert personal chef. 
    Analyze the provided photo of a refrigerator and identify the available ingredients, assigning a confidence level (high/medium/low) to each.
    Based on those ingredients, recommend 3 practical recipes that maximize ingredient usage and minimize additional purchases. 
    Use web search to find current recipes and food trends related to the available ingredients, prioritizing recent and reliable sources.
    You can also use available tools to enhance your recommendations.

    Rules:
    - Never invent ingredients that cannot reasonably be identified in the image.
    - Clearly distinguish between ingredients observed in the image and information obtained through web research.
    - Assume basic pantry staples (salt, pepper, oil, water) are available.
    - Formulate the entire response in Spanish.
    """

""" NO DENTRO DEL SYSTEM_PROMPT
    Response Format Example: 
    {{
        "recipes": {{
            "recipe1": "Texto completo de la receta...",
            "recipe2": "Texto completo de la receta...",
            "recipe3": "Texto completo de la receta..."
        }},
        "ingredients": ["salmón", "aguacate", "espinacas"]
    }} 
"""

# En el modulo 1.4 explica la forma de subir la imagen con Jupyter Notebooks, pero en el caso de VSCode no aplica.

image_path = Path(__file__).parent / "images" / "nevera.png"

if not image_path.is_file():
    raise FileNotFoundError(f"No existe la imagen: {image_path}")

mime_type = "image/png"

img_bytes = image_path.read_bytes()
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

multimodal_query = HumanMessage(content=[
    {"type": "text", "text": "This is the reference image"},
    {"type": "image", "base64": img_b64, "mime_type": "image/png"}
])

""" query = "Salmon, avocado, spniach, lettuce, tomato, onion, garlic" """

model = init_chat_model(
    model="gpt-5-nano",
    temperature=0.4
)

tavily_client = TavilyClient()

@tool("web_search")
def tool1(query: str) -> Dict[str, Any]:
    """ Updated web search for recipes and food trends."""
    return tavily_client.search(query)

agent = create_agent(
    model=model,
    tools=[tool1],
    system_prompt=system_prompt,
    # response_format=ResponseOutput,
    checkpointer=InMemorySaver()
)

config = {"configurable":{"thread_id":"1"}}

for token, metadata in agent.stream(
    {"messages": [multimodal_query]},
    stream_mode="messages",
    config=config,  # En Python los argumentos posicional deben ir siempre antes que los nombrados
):
    if token.content:
        print(token.content, end="", flush=True)

query = "¿Which of these recipes is your favorite?"

response = agent.invoke(
    {"messages": [HumanMessage(content=query)]},
    config,
)

print(response["messages"][-1].content)

""" pprint(response['messages']) """
