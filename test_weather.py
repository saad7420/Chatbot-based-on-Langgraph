import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

print(f"API Key Found: {bool(OPENWEATHER_API_KEY)}")

def test_weather_api(location: str):
    if not OPENWEATHER_API_KEY:
        print("Error: OPENWEATHER_API_KEY is not set in your .env file.")
        return

    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        print("\n--- Raw API Response Status ---")
        print(f"HTTP Status Code: {response.status_code}")
        
        if response.status_code == 200:
            weather = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            city = data["name"]
            country = data["sys"]["country"]
            
            print("\n--- Parsed Weather Output ---")
            print(f"Location: {city}, {country}")
            print(f"Condition: {weather.capitalize()}")
            print(f"Temperature: {temp}°C")
            print(f"Feels Like: {feels_like}°C")
            print("\nAPI Test Passed Successfully!")
        else:
            print(f"\nAPI Error Message: {data.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"\nConnection Error: {e}")

if __name__ == "__main__":
    # Test with a city name
    test_weather_api("London")