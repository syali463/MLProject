import os
import requests
import pandas as pd
import io
import time
import geopandas as gpd
import warnings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings('ignore')

API_KEY = "20F76B03-6A03-3D73-BFA1-C8C98B794FB8"
TARGET_CROP = "CORN"
STATE = "TX"
START_YEAR = 2000

NASA_PARAMS = "T2M_MAX,T2M_MIN,T2M,PRECTOTCORR,RH2M,T2MDEW,QV2M,ALLSKY_SFC_SW_DWN,WS2M"
GROWING_MONTHS = [3, 4, 5, 6, 7, 8]
NASA_FILE = "NASA_MASTER_WEATHER.csv"

session = requests.Session()
# Highly aggressive retry logic
retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

def get_texas_county_centroids():
    print("Mapping Texas County Centroids...")
    us_counties = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_20m.zip")
    texas = us_counties[us_counties['STATEFP'] == '48'].copy()
    texas['County'] = texas['NAME'].str.upper()
    texas_proj = texas.to_crs(epsg=3857)
    texas['centroid'] = texas_proj.centroid.to_crs(texas.crs)
    texas['LAT'] = texas['centroid'].y
    texas['LON'] = texas['centroid'].x
    return texas[['County', 'LAT', 'LON']]

def fetch_soil_data(lat, lon):
    url = f"https://rest.soilgrids.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=phh2o&property=clay&property=soc&depth=0-15cm"
    try:
        res = session.get(url, timeout=10).json()
        props = res.get('properties', {}).get('layers', [])
        soil_dict = {'Soil_pH': None, 'Soil_Clay_Pct': None, 'Soil_Organic_Carbon': None}
        for layer in props:
            name = layer.get('name')
            val = layer.get('depths', [{}])[0].get('values', {}).get('mean')
            if name == 'phh2o' and val: soil_dict['Soil_pH'] = val / 10.0
            elif name == 'clay' and val: soil_dict['Soil_Clay_Pct'] = val / 10.0
            elif name == 'soc' and val: soil_dict['Soil_Organic_Carbon'] = val / 10.0
        return soil_dict
    except Exception:
        return {'Soil_pH': None, 'Soil_Clay_Pct': None, 'Soil_Organic_Carbon': None}

def build_environmental_dataset(target_counties, target_years):
    print("Starting Resumable NASA Weather Extraction...")
    centroids = get_texas_county_centroids()
    
    # Check what we already have to avoid starting over
    completed_counties = []
    if os.path.exists(NASA_FILE):
        try:
            existing_df = pd.read_csv(NASA_FILE)
            if not existing_df.empty and 'County' in existing_df.columns:
                completed_counties = existing_df['County'].unique().tolist()
                print(f"Found existing file. Resuming... Already completed {len(completed_counties)} counties.")
        except Exception as e:
            print("Could not read existing file, starting fresh.")
            
    active_counties = centroids[centroids['County'].isin(target_counties)]
    
    # Determine if we need to write the CSV header
    write_header = not os.path.exists(NASA_FILE) or len(completed_counties) == 0

    for _, row in active_counties.iterrows():
        county = row['County']
        
        # Skip if we already pulled this county successfully
        if county in completed_counties:
            continue
            
        lat, lon = round(row['LAT'], 4), round(row['LON'], 4)
        print(f"Pulling data for {county}...")
        
        soil_metrics = fetch_soil_data(lat, lon)
        nasa_url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters={NASA_PARAMS}&community=AG&longitude={lon}&latitude={lat}&start={min(target_years)}&end={max(target_years)}&format=JSON"
        
        try:
            res = session.get(nasa_url, timeout=20)
            if res.status_code != 200:
                print(f"NASA API returned status {res.status_code} for {county}. Skipping for now.")
                time.sleep(2)
                continue
                
            res_json = res.json()
            weather_data = res_json.get('properties', {}).get('parameter', {})
            
            county_rows = []
            for year in target_years:
                row_data = {'Year': year, 'County': county}
                row_data.update(soil_metrics)
                for param, monthly_values in weather_data.items():
                    for month in GROWING_MONTHS:
                        date_key = f"{year}{month:02d}"
                        val = monthly_values.get(date_key)
                        row_data[f"M{month}_{param}"] = val if val != -999.0 else None
                county_rows.append(row_data)
                
            # Save this specific county to the file instantly
            df_temp = pd.DataFrame(county_rows)
            df_temp.to_csv(NASA_FILE, mode='a', index=False, header=write_header)
            write_header = False # Only write the header once
            
            completed_counties.append(county)
            print(f"SUCCESS: Saved {county}.")
            
        except Exception as e:
            print(f"ERROR on {county}: {e}. Moving to next.")
            
        # Mandatory sleep to prevent immediate IP ban from NASA
        time.sleep(1.5)

    print(f"\nExtraction routine finished. Data saved in {NASA_FILE}.")

if __name__ == "__main__":
    try:
        # Load the USDA file you already generated to get the target counties and years
        df_usda = pd.read_csv("USDA_MASTER_CORN.csv")
        target_counties = df_usda['County'].unique()
        target_years = sorted(df_usda['Year'].unique())
        build_environmental_dataset(target_counties, target_years)
    except FileNotFoundError:
        print("ERROR: USDA_MASTER_CORN.csv not found. Please ensure it is in the same folder.")