import requests
import pandas as pd
from bs4 import BeautifulSoup
import time 

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
    Fetches the HTML content and specifically reads the team stats table.
    """
    html_content = get_html_content(url)
    
    if html_content:
        try:
            # CORRECTED LINE: Using the id you provided
            df_list = pd.read_html(html_content, attrs={'id': 'stats_squads_standard_for'})
            df = df_list[0]
            print("Successfully read the team stats table.")
            return df
        except ValueError as e:
            print(f"No table with id 'stats_squads_standard_for' found on {url}. Error: {e}")
            return None
    return None

# The URL for the 2024-2025 Premier League stats
url = "https://fbref.com/en/comps/9/2024-2025/2024-2025-Premier-League-Stats"

# Call the function to get your DataFrame
team_stats_df = get_team_stats(url)

if team_stats_df is not None:
    print(team_stats_df.head())


# ... (your existing code) ...

def get_team_stats(url):
    # ... (code to read the table) ...
    if team_stats_df is not None:
        # Step 4: Flatten the multi-level headers
        # Use a list comprehension to combine the two levels
        new_columns = ['_'.join(col).strip() for col in team_stats_df.columns.values]
        
        # Assign the new column names to the DataFrame
        team_stats_df.columns = new_columns

        print("Flattened headers:")
        print(team_stats_df.head())
        
        return team_stats_df
