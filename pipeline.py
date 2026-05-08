import requests
import pandas as pd
import io
import time
import geopandas as gpd
from shapely.geometry import Point
import warnings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
API_KEY = "20F76B03-6A03-3D73-BFA1-C8C98B794FB8"
TARGET_CROP = "CORN"
STATE = "TX"
START_YEAR = 2000

# NASA Parameters based on XGBoost Literature
NASA_PARAMS = "T2M_MAX,T2M_MIN,T2M,PRECTOTCORR,RH2M,T2MDEW,QV2M,ALLSKY_SFC_SW_DWN,WS2M"
GROWING_MONTHS = [3, 4, 5, 6, 7, 8]

# Robust session handling to prevent random disconnects
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# ==========================================
# 1. USDA DATA (Yield & Planting Progress)
# ==========================================
def build_usda_dataset():
    print("Fetching USDA Yield data...")
    url_yield = f"https://quickstats.nass.usda.gov/api/api_GET/?key={API_KEY}&commodity_desc={TARGET_CROP}&statisticcat_desc=YIELD&state_alpha={STATE}&year__GE={START_YEAR}&format=CSV"
    res_yield = session.get(url_yield)
    df_yield_raw = pd.read_csv(io.StringIO(res_yield.text), low_memory=False)
    
    df_yield = df_yield_raw[
        (df_yield_raw['agg_level_desc'] == 'COUNTY') & 
        (df_yield_raw['unit_desc'] == 'BU / ACRE') & 
        (df_yield_raw['prodn_practice_desc'] == 'ALL PRODUCTION PRACTICES')
    ].copy()
    
    df_yield = df_yield[['year', 'county_name', 'Value']]
    df_yield.columns = ['Year', 'County', 'Yield_Bu_Acre']
    
    # Clean numeric values (remove commas just in case)
    df_yield['Yield_Bu_Acre'] = df_yield['Yield_Bu_Acre'].astype(str).str.replace(',', '')
    df_yield['Yield_Bu_Acre'] = pd.to_numeric(df_yield['Yield_Bu_Acre'], errors='coerce')
    df_yield = df_yield.dropna(subset=['Yield_Bu_Acre'])
    df_yield['County'] = df_yield['County'].str.upper()

    print("Fetching USDA Progress data (PCT PLANTED)...")
    url_prog = f"https://quickstats.nass.usda.gov/api/api_GET/?key={API_KEY}&commodity_desc={TARGET_CROP}&statisticcat_desc=PROGRESS&unit_desc=PCT PLANTED&state_alpha={STATE}&year__GE={START_YEAR}&format=CSV"
    res_prog = session.get(url_prog)
    df_prog_raw = pd.read_csv(io.StringIO(res_prog.text), low_memory=False)
    
    df_prog = df_prog_raw[['year', 'week_ending', 'Value']].copy()
    df_prog.columns = ['Year', 'Week Ending', 'Value']
    df_prog['Value'] = pd.to_numeric(df_prog['Value'], errors='coerce')
    df_prog = df_prog.dropna(subset=['Value'])

    df_prog['Week Ending'] = pd.to_datetime(df_prog['Week Ending'])
    df_prog['Month'] = df_prog['Week Ending'].dt.month_name()
    target_months = ['February', 'March', 'April', 'May', 'June'] 
    
    df_prog = df_prog[df_prog['Month'].isin(target_months)]
    df_monthly_prog = df_prog.groupby(['Year', 'Month'])['Value'].max().unstack().reset_index()
    df_monthly_prog = df_monthly_prog.rename(columns={m: f"PCT_PLANTED_{m}" for m in target_months if m in df_monthly_prog.columns})
    df_monthly_prog = df_monthly_prog.ffill(axis=1).fillna(0)

    # Merge Yield and Progress
    df_final = pd.merge(df_yield, df_monthly_prog, on='Year', how='inner')
    print("Saving USDA_MASTER_CORN.csv...")
    df_final.to_csv("USDA_MASTER_CORN.csv", index=False)
    return df_final

# ==========================================
# 2. ENVIRONMENTAL DATA (Weather & Soil)
# ==========================================
def get_texas_county_centroids():
    print("Mapping Texas County Centroids (applying accurate CRS projection)...")
    us_counties = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_20m.zip")
    texas = us_counties[us_counties['STATEFP'] == '48'].copy()
    texas['County'] = texas['NAME'].str.upper()
    
    # Proper geographic CRS math for accurate centroids
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
    print("Fetching NASA Weather and Soil Data...")
    centroids = get_texas_county_centroids()
    active_counties = centroids[centroids['County'].isin(target_counties)]
    env_data = []

    for _, row in active_counties.iterrows():
        county = row['County']
        lat, lon = round(row['LAT'], 4), round(row['LON'], 4)
        soil_metrics = fetch_soil_data(lat, lon)
        
        nasa_url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters={NASA_PARAMS}&community=AG&longitude={lon}&latitude={lat}&start={min(target_years)}&end={max(target_years)}&format=JSON"
        
        try:
            res = session.get(nasa_url, timeout=15).json()
            weather_data = res.get('properties', {}).get('parameter', {})
            for year in target_years:
                row_data = {'Year': year, 'County': county}
                row_data.update(soil_metrics)
                for param, monthly_values in weather_data.items():
                    for month in GROWING_MONTHS:
                        date_key = f"{year}{month:02d}"
                        val = monthly_values.get(date_key)
                        row_data[f"M{month}_{param}"] = val if val != -999.0 else None
                env_data.append(row_data)
        except Exception as e:
            print(f"Skipping weather for {county} due to API error: {e}")
            continue
        time.sleep(0.5)

    # Safe dropna so we don't nuke good data unnecessarily 
    df_env = pd.DataFrame(env_data).dropna()
    print("Saving NASA_MASTER_WEATHER.csv...")
    df_env.to_csv("NASA_MASTER_WEATHER.csv", index=False)
    return df_env

# ==========================================
# EXECUTE PIPELINE
# ==========================================
if __name__ == "__main__":
    df_usda = build_usda_dataset()
    if df_usda is not None:
        build_environmental_dataset(df_usda['County'].unique(), df_usda['Year'].unique())
        print("\nPipeline Complete. Data saved to USDA_MASTER_CORN.csv and NASA_MASTER_WEATHER.csv. Ready for XGBoost.")