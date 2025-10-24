from langchain_groq import ChatGroq
import os
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pprint import pprint
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState,StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from dotenv import load_dotenv


load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENWEATHERMAP_API_KEY"] = os.getenv("OPENWEATHERMAP_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

class city_info(BaseModel):
    Restraunt:str=Field(description="Best rated restraunts in the destination",min_length=50,max_length=1000)
    Activity:str=Field(description="Best rated activity in the destination",min_length=50,max_length=1000)
    Transport:str=Field(description="Convienient transport in the destination",min_length=10,max_length=1000)
    Attraction:str=Field(description="Attractions in the destination",min_length=50,max_length=1000)
    Cost:float=Field(description="Cost of travelling")
class HotelInfo(BaseModel):
    name: str = Field(description="Name of the hotel")
    cost_per_night: int = Field(description="Approximate cost per night in INR")
    location: str = Field(description="Area or landmark near hotel")
class Hotels(BaseModel):
    hotels: list[HotelInfo] = Field(description="List of top rated hotels in the given city")


class Destination:
    def __init__(self,city):
        self.city=city

class Itinerary(Destination):
    def __init__(self,city,city_info: city_info,Hotels: Hotels):
        self.city_info=city_info
        self.Hotels=Hotels
        super().__init__(city)
    
    def attractions(self):
        llm = ChatGroq(model_name="openai/gpt-oss-20b",temperature=0)
        template="""You are a travel assistant whose job is to tell the most famous attractions, best rated restraunts, activities,and convenient transport with cost in the destination given by the user.
                Do not breif the spots just tell the names and what they are good at.Tell 5 of every attraction,activty, and restraunt and 2 transport.Also give the total average cost of travelling there.
                Return the following
                - Restraunt
                - Activity
                - Transport
                - Attraction
                - Cost
                \n{format_instructions}\n{city}"""

        parser=JsonOutputParser(pydantic_object=city_info)
        prompt=PromptTemplate(
            template=template,
            input_variables=["city"],
            partial_variables={"format_instructions":parser.get_format_instructions()}
        )
        chain = prompt | llm | parser
        response=chain.invoke({"city":self.city})
        return response
    def weather(self):
        weather = OpenWeatherMapAPIWrapper()
        weather_data = weather.run(self.city)
        return weather_data
    def hotels(self):
        llm = ChatGroq(model_name="openai/gpt-oss-20b",temperature=0)
        template="""You are a travel assistant whose job is to tell the 5-6 best rated and budget friendly hotels in the city given by the user with costs.
        Do not breif the hotels just tell the rating,costs and where the location is.
        For each hotel, return:
        - name
        - cost_per_night (only integer number, in INR)
        - location
        \n{format_instructions}\n{city}"""

        parser=JsonOutputParser(pydantic_object=Hotels)
        prompt=PromptTemplate(
            template=template,
            input_variables=["city"],
            partial_variables={"format_instructions":parser.get_format_instructions()}
        )
        chain = prompt | llm | parser
        response=chain.invoke({"city":self.city})
        return response   
    

def summarize(results):
  llm = ChatGroq(model_name="openai/gpt-oss-20b",temperature=0)
  template="""You are a travel assistant whose job is to summarize the locations given and make a 5 day plan for the city given by the user. Dont do extra just summarize and give.
  You will get:
  - Activity
  - restraunt (only integer number)
  - attractions
  - transport
  - cost
  and you will return:
  A plan. Divide these things into 5 days.\n{input}"""

  parser=StrOutputParser()
  prompt=PromptTemplate(
      template=template,
      input_variables=["input"]
  )
  chain = prompt | llm | parser
  response=chain.invoke({"input":results})
  return response

from langchain.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The product of a and b.
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of a and b.
    """
    return a + b

@tool
def sub(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of a and b.
    """
    return a - b

@tool
def divide(a: int, b: int) -> float:
    """
    Divide two integers.

    Args:
        a (int): The numerator.
        b (int): The denominator (must not be 0).

    Returns:
        float: The result of division.
    """
    if b == 0:
        raise ValueError("Denominator cannot be zero.")
    return a / b


def calculator():
    llm = ChatGroq(model_name="openai/gpt-oss-20b",temperature=0)

    tools=[multiply, add, divide]
    llm_with_tools=llm.bind_tools(tools)

    def function_1(state:MessagesState):
        user_question=state["messages"]
        SYSTEM_PROMPT="You are a helpful assistant tasked with performing arithmetic on a set of inputs for the user to calculate their trip budget."
        input_question = [SYSTEM_PROMPT]+user_question
        response = llm_with_tools.invoke(input_question)
        return {
            "messages":[response]
        }

    cal=StateGraph(MessagesState)
    cal.add_node("llm_decision_step",function_1)
    cal.add_node("tools",ToolNode(tools))
    cal.add_edge(START,"llm_decision_step")
    cal.add_conditional_edges(
        "llm_decision_step",
        tools_condition,
    )
    cal.add_edge("tools","llm_decision_step")
    react_graph=cal.compile()
    return react_graph

