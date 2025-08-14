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

            # Drop the columns we have decided to not include
            columns_to_drop = [
                '# Pl', 
                'Age',
                'Playing Time_MP', 
                'Playing Time_Starts', 
                'Playing Time_Min', 
                'Playing Time_90s', 
                'Shooting_xG/Sh',
                'Shooting_SoT%',
                'Expected_npxG+xAG',
                'Per 90 Minutes_Gls',
                'Per 90 Minutes_Ast',
                'Per 90 Minutes_G+A',
                'Per 90 Minutes_G-PK',
                'Per 90 Minutes_G+A-PK',
                'Per 90 Minutes_xG',
                'Per 90 Minutes_xAG',
                'Per 90 Minutes_xG+xAG',
                'Per 90 Minutes_npxG',
                'Per 90 Minutes_npxG+xAG'
            ]
            
            # Drop the columns if they exist in the DataFrame
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

            return df
        except ValueError as e:
            print(f"No table with id 'stats_squads_standard_for' found on {url}. Error: {e}")
            return None
    return None

# --- Main script execution starts here ---

# A list of years to scrape
years = range(2017, 2025)

# The base URL, with a placeholder for the year.
base_url = "https://fbref.com/en/comps/9/{year}-{next_year}/Premier-League-Stats"

for year in years:
    # Construct the URL for the specific year
    next_year = year + 1
    url = base_url.format(year=year, next_year=next_year)
    print(f"\n--- Scraping data for the {year}-{next_year} season from {url} ---")
    
    # Call the function to get your DataFrame
    team_stats_df = get_team_stats(url)

    if team_stats_df is not None:
        print("Final DataFrame head:")
        print(team_stats_df.head())
        
        # Define the output file name with the year
        output_path = f'footballprediction/fbref_data/scrapped_team_stats_{year}.csv'
        team_stats_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved the scraped data to '{output_path}'")
    else:
        print(f"Could not scrape data for the {year}-{next_year} season.")