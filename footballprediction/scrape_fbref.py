import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
import time # Import the time module

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

def scrape_table(html_content, table_id):
    """
    Scrapes a single table from the HTML content and returns a DataFrame.
    """
    try:
        # Use pandas to read the table with the specified ID
        df_list = pd.read_html(StringIO(html_content), attrs={'id': table_id}, header=[0, 1])
        df = df_list[0]
        # Drop the redundant header rows if they exist in the table body (e.g., "Squad" repeated)
        if 'Squad' in df.columns:
            df = df.drop(df[df['Squad'] == 'Squad'].index)
        return df
    except ValueError as e:
        print(f"No table with id '{table_id}' found. Error: {e}")
        return None

def clean_and_prefix_df(df, prefix):
    """
    Flattens multi-level headers and applies a prefix to ALL columns except 'Squad'.
    This ensures distinct names for metrics that appear in both offensive and defensive tables.
    """
    if df is not None:
        # First, flatten the multi-level headers (e.g., ('Playing Time', 'MP') -> 'Playing Time_MP')
        flattened_cols = []
        for top_level, sub_level in df.columns:
            if "Unnamed" in top_level: # Covers 'Squad', 'Rk' sometimes if they are standalone headers
                flattened_cols.append(sub_level)
            else:
                flattened_cols.append(f"{top_level}_{sub_level}")
        df.columns = flattened_cols

        # Now, apply the prefix to all columns except 'Squad'
        new_columns = []
        for col in df.columns:
            if col == 'Squad': # 'Squad' is the unique identifier for merging
                new_columns.append(col)
            else:
                new_columns.append(f"{prefix}_{col}")
        df.columns = new_columns
        return df
    return None


def get_and_process_stats(url):
    """
    Fetches, scrapes, cleans, and merges data from both offensive and defensive tables.
    """
    html_content = get_html_content(url)
    if not html_content:
        return None

    # Scrape and process the offensive stats table ('stats_squads_standard_for')
    df_offensive_raw = scrape_table(html_content, 'stats_squads_standard_for')
    df_offensive = clean_and_prefix_df(df_offensive_raw.copy(), 'Offensive')

    # Scrape and process the defensive stats table ('stats_squads_standards_against')
    df_defensive_raw = scrape_table(html_content, 'stats_squads_standard_against')
    df_defensive = clean_and_prefix_df(df_defensive_raw.copy(), 'Defensive')
    
    if df_offensive is None or df_defensive is None:
        print("One or more tables could not be scraped. Returning None.")
        return None

    # Ensure 'Squad' column is present in both before merge
    if 'Squad' not in df_offensive.columns or 'Squad' not in df_defensive.columns:
        print("Error: 'Squad' column missing from one of the DataFrames. Cannot merge.")
        return None

    # Merge the two DataFrames into a single one on the 'Squad' column
    df_combined = pd.merge(df_offensive, 
                           df_defensive, # Merge all columns from the defensive table
                           on='Squad', 
                           how='left')
    
    # Reset the index after dropping rows if any were removed
    df_combined = df_combined.reset_index(drop=True)

    return df_combined

# --- Main script execution starts here ---

# A list of years to scrape
years = range(2017, 2025)

# The base URL, with a placeholder for the year.
base_url = "https://fbref.com/en/comps/9/{year}-{next_year}/Premier-League-Stats"

for year in years:
    next_year = year + 1
    url = base_url.format(year=year, next_year=next_year)
    print(f"\n--- Scraping data for the {year}-{next_year} season from {url} ---")
    
    team_stats_df = get_and_process_stats(url)

    if team_stats_df is not None:
        print(f"DataFrame for {year}-{next_year} season has {len(team_stats_df.columns)} columns.")
        print("Final DataFrame head:")
        print(team_stats_df.head())
        print("All columns in the final DataFrame:")
        print(team_stats_df.columns.tolist())
        
        output_path = f'footballprediction/fbref_data/scrapped_team_stats_{year}.csv'
        team_stats_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved the scraped data to '{output_path}'")
    else:
        print(f"Could not scrape data for the {year}-{next_year} season.")

    # --- Add a longer delay to avoid rate limiting ---
    time.sleep(15) # Pause for 15 seconds between requests to different seasons

# Note: The time delay is increased to 15 seconds to reduce the risk of being blocked by the server.
# Adjust this value as needed based on the server's response and your scraping needs.