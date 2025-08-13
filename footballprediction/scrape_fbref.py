import requests
import pandas as pd
from bs4 import BeautifulSoup
import time 
from io import StringIO

def get_html_content(url):
    """
    Fetches the HTML content of a given URL.
    Includes headers to mimic a web browser.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.text
    else:
        print(f"Error fetching {url}: Status Code {response.status_code}")
        return None

def get_team_stats(url):
    """
    Fetches the HTML content, reads the team stats table, and cleans the headers.
    """
    html_content = get_html_content(url)
    
    if html_content:
        try:
            df_list = pd.read_html(StringIO(html_content), attrs={'id': 'stats_squads_standard_for'}, header=[0, 1])
            df = df_list[0]
            print("Successfully read the team stats table.")

            # This is the most robust and explicit way to flatten the headers for this table
            df.columns = [f"{top}_{bottom}" if "Unnamed" not in top else bottom for top, bottom in df.columns]

            # Now, drop any rows that are just header rows repeated in the table body.
            df = df.drop(df[df['Squad'] == 'Squad'].index)
            
            # Reset the index after dropping rows
            df = df.reset_index(drop=True)
            
            # Removed the debugging print statement from here.
            
            return df
        except ValueError as e:
            print(f"No table with id 'stats_squads_standard_for' found on {url}. Error: {e}")
            return None
    return None

# --- Main script execution starts here ---

# The URL for the 2024-2025 Premier League stats
url = "https://fbref.com/en/comps/9/2024-2025/2024-2025-Premier-League-Stats"

# Call the function to get your DataFrame
team_stats_df = get_team_stats(url)

if team_stats_df is not None:
    print("\nFinal DataFrame head:")
    print(team_stats_df.head())