"""
Tools for EcoHome Energy Advisor Agent
"""

import os
import json
import random
import requests
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models.energy import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager()


@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast for a specific location and number of days.

    Args:
        location (str): Location to get weather for (e.g., "San Francisco, CA")
        days (int): Number of days to forecast (1-7)

    Returns:
        Dict[str, Any]: Weather forecast data for a given location and time range.

        Structure:
        forecast = {
            "location": str,            # Human-readable location (e.g. "Tokyo, Japan")
            "forecast_days": int,        # Number of forecast days returned
            "current": {
                "temperature_c": float,  # Current air temperature at 2 m height (°C)
                "condition": str,        # Derived condition (e.g. "sunny", "cloudy")
                "cloud_cover": int,      # Cloud cover in percent (0–100)
                "solar_irradiance": float,  # Global shortwave radiation (W/m²)
                "humidity": int,         # Relative humidity at 2 m height (%)
                "wind_speed": float      # Wind speed at 10 m height (km/h)
            },
            "units": {
                "time": "iso8601",           # Timestamp format
                "temperature_2m": "°C",
                "relative_humidity_2m": "%",
                "wind_speed_10m": "km/h",
                "shortwave_radiation": "W/m²",
                "cloud_cover": "%"
            },
            "hourly": [
                {
                    "time": str,             # ISO 8601 timestamp (local time)
                    "temperature_c": float,  # Air temperature at 2 m height (°C)
                    "condition": str,        # Derived condition for the hour
                    "cloud_cover": int,      # Cloud cover in percent (0–100)
                    "solar_irradiance": float,  # Global shortwave radiation (W/m²)
                    "humidity": int,         # Relative humidity at 2 m height (%)
                    "wind_speed": float      # Wind speed at 10 m height (km/h)
                },
                ...
            ]
        }
    """
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1},
        timeout=10,
    ).json()

    if not geo.get("results"):
        raise ValueError(f"Location not found: {location}")

    loc = geo["results"][0]
    lat, lon = loc["latitude"], loc["longitude"]

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "shortwave_radiation",
                "cloud_cover",
            ],
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "shortwave_radiation",
                "cloud_cover",
            ],
            "forecast_days": days,
            "timezone": "auto",
        },
        timeout=10,
    ).json()

    hourly = weather["hourly"]
    hours = min(days * 24, len(hourly["temperature_2m"]))
    forecast = {
        "location": f"{loc['name']}, {loc['country']}",
        "forecast_days": days,
        "current": {
            "temperature_c": weather["current"]["temperature_2m"],
            "condition": "cloudy"
            if weather["current"]["cloud_cover"] > 50
            else "sunny",
            "cloud_cover": weather["current"]["cloud_cover"],
            "solar_irradiance": weather["current"]["shortwave_radiation"],
            "humidity": weather["current"]["relative_humidity_2m"],
            "wind_speed": weather["current"]["wind_speed_10m"],
        },
        "units": weather["hourly_units"],
        "hourly": [],
    }
    for i in range(hours):
        forecast["hourly"].append(
            {
                "time": hourly["time"][i],
                "temperature_c": hourly["temperature_2m"][i],
                "condition": "cloudy" if hourly["cloud_cover"][i] > 50 else "sunny",
                "cloud_cover": hourly["cloud_cover"][i],
                "solar_irradiance": hourly["shortwave_radiation"][i],
                "humidity": hourly["relative_humidity_2m"][i],
                "wind_speed": hourly["wind_speed_10m"][i],
            }
        )

    return forecast


# Implement get_electricity_prices tool
@tool
def get_electricity_prices(date: str = None) -> Dict[str, Any]:
    """
    Get electricity prices for a specific date or current day.

    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)

    Returns:
        Dict[str, Any]: Electricity pricing data with hourly rates
        E.g:
        prices = {
            "date": ...,
            "pricing_type": "time_of_use",
            "currency": "USD",
            "unit": "per_kWh",
            "hourly_rates": [
                {
                    "hour": .., # for hour in range(24)
                    "rate": ..,
                    "period": ..,
                    "demand_charge": ...
                }
            ]
        }
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # Mock electricity pricing - in real implementation, this would call a pricing API
    # Use a base price per kWh
    # Then generate hourly rates with peak/off-peak pricing
    # Peak normally between 6 and 22...
    # demand_charge should be 0 if off-peak

    peak_hours = list(range(6, 22))

    return {
        "date": date,
        "pricing_type": "time_of_use",
        "currency": "USD",
        "unit": "per_kWh",
        "hourly_rates": [
            {
                "hour": i,
                "rate": 0.18 if i in peak_hours else 0.15,
                "period": "peak" if i in peak_hours else "off peak",
                "demand_charge": 1 if i in peak_hours else 0,
            }
            for i in range(24)
        ],
    }


@tool
def query_energy_usage(
    start_date: str, end_date: str, device_type: str = None
) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")

    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        records = db_manager.get_usage_by_date_range(start_dt, end_dt)

        if device_type:
            records = [r for r in records if r.device_type == device_type]

        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": [],
        }

        for record in records:
            usage_data["records"].append(
                {
                    "timestamp": record.timestamp.isoformat(),
                    "consumption_kwh": record.consumption_kwh,
                    "device_type": record.device_type,
                    "device_name": record.device_name,
                    "cost_usd": record.cost_usd,
                }
            )

        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}


@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        records = db_manager.get_generation_by_date_range(start_dt, end_dt)

        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(
                sum(r.generation_kwh for r in records)
                / max(1, (end_dt - start_dt).days),
                2,
            ),
            "records": [],
        }

        for record in records:
            generation_data["records"].append(
                {
                    "timestamp": record.timestamp.isoformat(),
                    "generation_kwh": record.generation_kwh,
                    "weather_condition": record.weather_condition,
                    "temperature_c": record.temperature_c,
                    "solar_irradiance": record.solar_irradiance,
                }
            )

        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}


@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.

    Args:
        hours (int): Number of hours to look back (default 24)

    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)

        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(
                    sum(r.consumption_kwh for r in usage_records), 2
                ),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {},
            },
            "generation": {
                "total_generation_kwh": round(
                    sum(r.generation_kwh for r in generation_records), 2
                ),
                "average_weather": "sunny" if generation_records else "unknown",
            },
        }

        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0,
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += (
                record.consumption_kwh
            )
            summary["usage"]["device_breakdown"][device]["cost_usd"] += (
                record.cost_usd or 0
            )
            summary["usage"]["device_breakdown"][device]["records"] += 1

        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)

        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}


@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.

    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return

    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    try:
        # Initialize vector store if it doesn't exist
        persist_directory = "data/vectorstore"
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        # Load documents if vector store doesn't exist
        if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
            # Load documents
            documents = []
            for doc_path in [
                "data/documents/tip_device_best_practices.txt",
                "data/documents/tip_energy_savings.txt",
            ]:
                if os.path.exists(doc_path):
                    loader = TextLoader(doc_path)
                    docs = loader.load()
                    documents.extend(docs)

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            # Create vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
        else:
            # Load existing vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma(
                persist_directory=persist_directory, embedding_function=embeddings
            )

        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=max_results)

        results = {"query": query, "total_results": len(docs), "tips": []}

        for i, doc in enumerate(docs):
            results["tips"].append(
                {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "relevance_score": "high"
                    if i < 2
                    else "medium"
                    if i < 4
                    else "low",
                }
            )

        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}


@tool
def calculate_energy_savings(
    device_type: str,
    current_usage_kwh: float,
    optimized_usage_kwh: float,
    price_per_kwh: float = 0.12,
) -> Dict[str, Any]:
    """
    Calculate potential energy savings from optimization.

    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)

    Returns:
        Dict[str, Any]: Savings calculation results
    """
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (
        (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0
    )

    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(savings_usd * 365, 2),
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings,
]
