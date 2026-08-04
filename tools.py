import os
import datetime
import requests
from dotenv import load_dotenv
from typing import Literal

from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment or .env file.")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.1
)

# -------------------------------------------------------------------
# Input Schemas
# -------------------------------------------------------------------

class CalculatorInput(BaseModel):
    first_num: float = Field(description="First numerical value")
    second_num: float = Field(description="Second numerical value")
    operation: Literal["add", "sub", "mul", "div"] = Field(description="Arithmetic operation to perform")

class StockInput(BaseModel):
    symbol: str = Field(description="Stock ticker symbol, e.g. AAPL or TSLA")

class WeatherInput(BaseModel):
    location: str = Field(description="City or region name, e.g. London")

class UnitConvertInput(BaseModel):
    value: float = Field(description="Numeric quantity to convert")
    from_unit: str = Field(description="Starting unit, e.g. km, kg, c, miles, lbs, f")
    to_unit: str = Field(description="Target unit, e.g. miles, lbs, f, km, kg, c")

class MapInput(BaseModel):
    location_query: str = Field(description="Address, place name, or region to locate")

class EmailInput(BaseModel):
    recipient_email: str = Field(description="Target email address")
    subject: str = Field(description="Subject line of the email")
    body: str = Field(description="Main body content of the email")

class CalendarInput(BaseModel):
    action: Literal["create", "search", "delete"] = Field(description="Action to execute")
    event_title: str = Field(description="Title or summary of the event")
    date_time: str = Field(description="Target date and time string")

class MemoryInput(BaseModel):
    action: Literal["save", "get"] = Field(description="Action to take on memory")
    key: str = Field(description="Identifier or category key")
    value: str = Field(default="", description="Value to store if action is save")

class QueryInput(BaseModel):
    query: str = Field(description="Search string or prompt")

# -------------------------------------------------------------------
# Tools Implementation
# -------------------------------------------------------------------

@tool(args_schema=QueryInput)
def web_search(query: str) -> str:
    """Search the web for real-time news, current events, or online information."""
    try:
        search = DuckDuckGoSearchRun(region='us-en')
        return search.run(query)
    except Exception as e:
        return f"Search failed: {e}"

@tool(args_schema=CalculatorInput)
def calculator(first_num: float, second_num: float, operation: str) -> str:
    """Perform basic arithmetic (add, sub, mul, div) on two numbers."""
    if operation == "add":
        res = first_num + second_num
    elif operation == "sub":
        res = first_num - second_num
    elif operation == "mul":
        res = first_num * second_num
    elif operation == "div":
        if second_num == 0:
            return "Error: Division by zero is impossible."
        res = first_num / second_num
    else:
        return f"Error: Invalid operation '{operation}'."
    return f"Result: {res}"

@tool(args_schema=StockInput)
def get_stock_price(symbol: str) -> str:
    """Fetch live market stock prices by ticker symbol."""
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=L7X0V36N67ZILY9R"
        response = requests.get(url, timeout=8)
        return response.text
    except Exception as e:
        return f"Error fetching stock data: {e}"

@tool(args_schema=WeatherInput)
def get_weather(location: str) -> str:
    """Fetch current weather metrics for a specified city or area."""
    if not OPENWEATHER_API_KEY:
        return "Missing OPENWEATHER_API_KEY environment variable."
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url, timeout=8).json()
        if res.get("cod") != 200:
            return f"Weather Error: {res.get('message', 'Failed request')}"
        
        desc = res["weather"][0]["description"]
        temp = res["main"]["temp"]
        feels = res["main"]["feels_like"]
        return f"Weather in {location}: {desc.capitalize()}, Temp: {temp}°C (Feels like: {feels}°C)."
    except Exception as e:
        return f"Weather connection error: {e}"

@tool
def get_current_datetime() -> str:
    """Return current date, timestamp, and day of the week."""
    now = datetime.datetime.now()
    return now.strftime("Current Date & Time: %Y-%m-%d %H:%M:%S (%A)")

@tool(args_schema=UnitConvertInput)
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert measurement units (km/miles, kg/lbs, celsius/fahrenheit)."""
    f, t = from_unit.lower().strip(), to_unit.lower().strip()
    if f in ["km", "kilometers"] and t in ["miles", "mi"]:
        return f"{value} km = {round(value * 0.621371, 2)} miles"
    elif f in ["miles", "mi"] and t in ["km", "kilometers"]:
        return f"{value} miles = {round(value * 1.60934, 2)} km"
    elif f in ["kg", "kilograms"] and t in ["lbs", "pounds"]:
        return f"{value} kg = {round(value * 2.20462, 2)} lbs"
    elif f in ["lbs", "pounds"] and t in ["kg", "kilograms"]:
        return f"{value} lbs = {round(value * 0.453592, 2)} kg"
    elif f in ["c", "celsius"] and t in ["f", "fahrenheit"]:
        return f"{value}°C = {round((value * 9/5) + 32, 2)}°F"
    elif f in ["f", "fahrenheit"] and t in ["c", "celsius"]:
        return f"{value}°F = {round((value - 32) * 5/9, 2)}°C"
    return f"Conversion from {from_unit} to {to_unit} not supported."

@tool(args_schema=MapInput)
def maps_tool(location_query: str) -> str:
    """Find map coordinates and display addresses using OpenStreetMap."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": location_query, "format": "json", "limit": 1}
        headers = {"User-Agent": "LangChainChatbot/1.0"}
        res = requests.get(url, params=params, headers=headers, timeout=8).json()
        if res:
            p = res[0]
            return f"Address: {p.get('display_name')}\nCoordinates: Lat {p.get('lat')}, Lon {p.get('lon')}"
        return f"No location found for '{location_query}'."
    except Exception as e:
        return f"Map lookup failed: {e}"

@tool(args_schema=EmailInput)
def send_email(recipient_email: str, subject: str, body: str) -> str:
    """Queue and send an email notification."""
    return f"Email queued to {recipient_email} with subject '{subject}'."

@tool(args_schema=CalendarInput)
def calendar_event_manager(action: str, event_title: str, date_time: str) -> str:
    """Create, search, or remove calendar events."""
    return f"Executed action '{action}' for calendar event '{event_title}' at {date_time}."

@tool(args_schema=MemoryInput)
def personal_memory_store(action: str, key: str, value: str = "") -> str:
    """Save or retrieve long-term user facts and context."""
    if action == "save":
        return f"Memory stored: [{key} = {value}]"
    elif action == "get":
        return f"Memory retrieved for key: '{key}'"
    return "Invalid memory command."

@tool(args_schema=QueryInput)
def search_conversation_history(query: str) -> str:
    """Search prior conversation logs and user thread history."""
    return f"Searched session history logs for: '{query}'."

# -------------------------------------------------------------------
# Tool Binding
# -------------------------------------------------------------------

tools = [
    get_stock_price,
    web_search,
    calculator,
    get_weather,
    get_current_datetime,
    unit_converter,
    maps_tool,
    send_email,
    calendar_event_manager,
    personal_memory_store,
    search_conversation_history
]

llm_with_tools = llm.bind_tools(tools)