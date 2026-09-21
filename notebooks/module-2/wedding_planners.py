from dotenv import load_dotenv
load_dotenv()

import asyncio
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


from langchain_community.utilities import SQLDatabase
db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")


orchestrator_system_prompt = f"""
    You are a Wedding Planner Orchestrator responsible for coordinating three specialized agents:
    travel_agent: finds round-trip flights to the wedding destination.
    venue_agent: finds suitable wedding venues or estates based on location, date, guests, budget, and preferences.
    dj_agent: searches a music database and creates a suitable wedding playlist.
    
    Your job is to:

    - Understand the user's requirements and extract relevant constraints.
    - Delegate each task to the appropriate specialized agent.
    - Pass all relevant context and constraints to each agent.
    - Coordinate multiple agents when necessary and keep their results consistent.
    - Combine their results into a clear, coherent wedding plan.
    - Never invent information. If essential information is missing, ask the user for clarification.
    - Act as the central coordinator: let each specialized agent handle its domain and focus on integrating their results into the best possible overall wedding plan.
    - Read and update state constraints when necessary.
"""

travel_agent_system_prompt = f"""
    You are a specialized Travel Agent responsible for finding suitable round-trip flights for a wedding.
    Use the available Kimi MCP flight search tool to find the best flight options based on the user's requirements.

    Your responsibilities:

    - Identify the departure airport, destination, travel dates, and number of travelers.
    - Search for suitable round-trip flights using the MCP tool.
    - Consider relevant constraints such as budget, duration, stops, and preferred times when provided.
    - Return the most relevant flight options with their key details, including airline, departure/arrival times, duration, stops, and price.
    - Never invent flight information. If essential information is missing, ask for it.

    Return concise, structured results that the Wedding Planner Orchestrator can easily integrate into the final wedding plan.
"""

venue_agent_system_prompt = f"""
    You are a specialized Venue Agent responsible for finding suitable wedding venues.
    Use the available VenuNite Events MCP tool to find the best venue options based on the user's requirements.

    Your responsibilities:

    - Identify the location, wedding date, number of guests, and budget.
    - Search for suitable wedding venues using the MCP tool.
    - Consider relevant constraints such as capacity, price, location, availability, and venue style when provided.
    - Return the most relevant venue options with their key details, including name, location, capacity, price, and relevant features.
    - Never invent venue information. If essential information is missing, ask for it.

    Return concise, structured results that the Wedding Planner Orchestrator can easily integrate into the final wedding plan.
"""

dj_agent_system_prompt = f"""
You are a specialized DJ Agent responsible for creating suitable wedding playlists.
Use the available SQL database tool to search for songs and build a playlist based on the user's requirements.

Your responsibilities:

- Identify the wedding style, musical preferences, and any relevant constraints.
- Search the SQL music database for suitable songs.
- Consider relevant factors such as genre, mood, artist, popularity, and requested preferences when provided.
- Return a well-balanced playlist with the song title, artist, and relevant details.
- Never invent songs or database information. If essential information is missing, ask for it.

Return concise, structured results that the Wedding Planner Orchestrator can easily integrate into the final wedding plan.
"""

travel_client = MultiServerMCPClient(
    {
        "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com",
                # "command": "python" solo vale para transport studio
            }
    }
)

venue_client = MultiServerMCPClient(
    {
        "venunite-events": {
            "transport": "streamable_http",
            "url": "https://mcp.venunite.com/v1/mcp/"
        }
    }
)

model = init_chat_model(
    model="gpt-5-nano",
    temperature=0.4
)

class CustomState(AgentState): 
    number_of_passengers: int
    cabin_class: str

@dataclass
class TripConstraints:
    trip_year: str = "2026"
    # number_of_passengers: int = 2
    # cabin_class: str = "Economy"
    paris_airport: str = "CDG"
    stops: int = 0
    baggage: str = "One big suitcase"
    flexibility: str = "+/- one day around"

@tool
def get_trip_year(runtime: ToolRuntime) -> str:
    """Get the year that the trips are going to take place"""
    return runtime.context.trip_year

@tool
def update_number_of_passengers(number_of_passengers: int, runtime: ToolRuntime) -> Command:
    """Update the number of passengers that are going to take the flights"""
    return Command(update={
        "number_of_passengers": number_of_passengers,
        "messages": [ToolMessage("Successfully updated number of passengers", tool_call_id=runtime.tool_call_id)]}
        )

@tool
def read_cabin_class(runtime: ToolRuntime) -> str:
    """Read the preferred cabin class of the passengers"""
    try:
        return runtime.state["cabin_class"]
    except KeyError:
        return "No preferred cabin class found"


async def wedding_planners():

    mcp_travel_tool = await travel_client.get_tools()

    travel_subagent = create_agent(
        model=model,
        tools=[mcp_travel_tool],
        system_prompt=travel_agent_system_prompt
    )

    @tool
    async def call_travel_subagent(query: str) -> str:
        """Call a travel subagent in order to find transport availability"""
        response = await travel_subagent.ainvoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

    mcp_venue_tool = await venue_client.get_tools()

    venue_subagent = create_agent(
        model=model,
        tools=[mcp_venue_tool],
        system_prompt=venue_agent_system_prompt
    )

    @tool
    async def call_venue_subagent(query: str) -> str:
        """Call a venue subagent in order to find venues availability"""
        response = await venue_subagent.ainvoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

    @tool
    def query_playlist_db(query: str) -> str:
        """Query the database for playlist information"""
        try:
            return db.run(query)
        except Exception as e:
            return f"Error querying database: {e}"
        
    dj_subagent = create_agent(
        model=model,
        tools=[query_playlist_db],
        system_prompt=dj_agent_system_prompt
    )

    @tool
    async def call_dj_subagent(query: str) -> str:
        """Call a dj subagent in order to make a playlist for the wedding"""
        response = await dj_subagent.ainvoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

    agent = create_agent(
        model=model,
        tools=[get_trip_year, update_number_of_passengers, read_cabin_class, call_travel_subagent],
        system_prompt=orchestrator_system_prompt,
        context_schema=TripConstraints,
        state_schema=CustomState,
        checkpointer=InMemorySaver() 
    )

    return agent


async def main():
    query = "Plan a wedding from "

    agent = await wedding_planners()

    config = {"configurable": {"thread_id": "3"}}

    response = await agent.ainvoke(
        {
            "messages": [HumanMessage(content=query)],
            "cabin_class": "Economy"
        },
        context=TripConstraints(),
        config=config
    )

    print(response["messages"][-1].content)


if __name__ == "__main__":  # Solo si se importa tendría el valor wedding_planners. Si se ejecuta este archivo Python define que tiene que ser __main__
    asyncio.run(main())

