import pandas as pd
import requests
import os
import pickle
from bs4 import BeautifulSoup
import time
import json
from datetime import datetime
# Import the Azure parser
from experiments.azure_parser import parse_azure_instance_data

# ----- DATA COLLECTION AND PREPARATION -----

# Directory for storing pickle files
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
os.makedirs(DATA_DIR, exist_ok=True)

def get_cached_data(provider):
    """
    Try to load data from pickle file
    Returns None if pickle doesn't exist or is older than 7 days
    """
    pickle_path = os.path.join(DATA_DIR, f"{provider.lower()}_instances.pkl")
    
    if os.path.exists(pickle_path):
        # Check if the pickle is fresh (less than 7 days old)
        mod_time = os.path.getmtime(pickle_path)
        if (time.time() - mod_time) < 7 * 24 * 60 * 60:  # 7 days in seconds
            try:
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading {provider} pickle: {e}")
    
    return None

def save_to_pickle(data, provider):
    """Save data to a pickle file"""
    pickle_path = os.path.join(DATA_DIR, f"{provider.lower()}_instances.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {provider} data to {pickle_path}")

def fetch_azure_instance_data():
    """
    Fetch Azure instance data from local HTML file
    Returns structured data for various instance types
    """
    # Try to load from cache first
    cached_data = get_cached_data('Azure')
    if cached_data is not None:
        print("Using cached Azure instance data")
        return cached_data
    
    print("Parsing Azure instance data from local HTML file...")
    try:

        # Use the parse_azure_instance_data function from azure_parser.py
        instances = parse_azure_instance_data("../data/azure.html")
        
        # If we couldn't get data, raise an exception
        if len(instances) == 0:
            raise Exception("No instance data found in Azure HTML")
        
        # Convert instance list to DataFrame
        # Map the fields appropriately
        df_data = []
        for instance in instances:
            df_data.append({
                "name": instance["name"],
                "vcpu": instance["vcpu"],
                "memory": instance["memory"],
                "storage": instance["storage"],
                "cost": instance["cost_hourly"]  # Use hourly cost from parser
            })
        
        df = pd.DataFrame(df_data)
        
        # Save to pickle for future use
        save_to_pickle(df, 'Azure')
        
        return df
        
    except Exception as e:
        print(f"Error parsing Azure data: {e}")
        # Create a minimal DataFrame to avoid errors
        return pd.DataFrame(columns=["name", "vcpu", "memory", "storage", "cost"])

def fetch_linode_instance_data():
    """
    Fetch Linode instance data from public source
    Returns structured data for various instance types
    """
    # Try to load from cache first
    cached_data = get_cached_data('Linode')
    if cached_data is not None:
        print("Using cached Linode instance data")
        return cached_data
    
    print("Fetching Linode instance data from public source...")
    # Linode API - public pricing data
    url = "https://api.linode.com/v4/linode/types"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    data = response.json()
    
    instances = []
    
    # Process the data from the API
    if 'data' in data:
        for item in data['data']:
            instance = {
                "name": item.get('label', ''),
                "vcpu": item.get('vcpus', 0),
                "memory": item.get('memory') / 1024 if 'memory' in item else 0,  # Convert MB to GB
                "storage": item.get('disk', 0),
                "cost": item.get('price', {}).get('hourly', 0)
            }
            instances.append(instance)
    
    # If we couldn't get data from the API, fallback to static data
    if len(instances) == 0:
        raise Exception("No data found from Linode API")
    
    df = pd.DataFrame(instances)
    
    # Save to pickle for future use
    save_to_pickle(df, 'Linode')
    
    return df

def prepare_cloud_data():
    """
    Prepare the integrated dataset from all cloud providers
    """
    
    azure_data = fetch_azure_instance_data()
    azure_data['provider'] = 'Azure'
    
    linode_data = fetch_linode_instance_data()
    linode_data['provider'] = 'Linode'
    
    # Combine all provider data
    all_data = pd.concat([ azure_data, linode_data], ignore_index=True)
    
    # Convert memory to GB for consistency (if needed)
    all_data['memory'] = all_data['memory'].apply(lambda x: x if x >= 1 else x * 1024)
    
    print("\n=== TOP 5 INSTANCES FROM AZURE ===")
    print(azure_data.head())
    
    print("\n=== TOP 5 INSTANCES FROM LINODE ===")
    print(linode_data.head())
    
    print("\n=== COMBINED DATA SAMPLE ===")
    print(all_data.head())
    
    # Save combined data to pickle
    save_to_pickle(all_data, 'all_providers')
    
    return all_data

# If this file is run directly, test the data collection
if __name__ == "__main__":
    print("Testing cloud data collection...")
    data = prepare_cloud_data()
    print(f"Collected data for {len(data)} instance types across all providers")
    print(data.head())
