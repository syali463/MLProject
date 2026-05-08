import pandas as pd
import requests
import geopandas as gpd
import time
import warnings
warnings.filterwarnings('ignore')

print("Loading USDA target counties...")
# Only pull soil for the counties we actually have yield data for
try:
    usda = pd.read_csv("USDA_MASTER_CORN.csv")
    target_counties = usda['County'].unique()
except FileNotFoundError:
    print("ERROR: USDA_MASTER_CORN.csv not found.")
    exit()

print("Mapping Texas County Centroids...")
us_counties = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_20m.zip")
texas = us_counties[us_counties['STATEFP'] == '48'].copy()
texas['County'] = texas['NAME'].str.upper()

texas_proj = texas.to_crs(epsg=3857)
texas['centroid'] = texas_proj.centroid.to_crs(texas.crs)
texas['LAT'] = texas['centroid'].y
texas['LON'] = texas['centroid'].x

active_counties = texas[texas['County'].isin(target_counties)]
soil_data_list = []

print(f"Beginning USDA SSURGO SQL Extraction for {len(active_counties)} counties...")

# The official USDA Soil Data Access API Endpoint
url = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"

for _, row in active_counties.iterrows():
    county = row['County']
    lat, lon = round(row['LAT'], 4), round(row['LON'], 4)
    
    # We write a raw SQL query that finds the map unit at our coordinate, 
    # grabs the dominant component, and pulls the clay, pH, and organic matter 
    # for the top 15cm horizon.
    query = f"""
    SELECT TOP 1 ch.claytotal_r, ch.ph1to1h2o_r, ch.om_r
    FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('POINT({lon} {lat})') AS p
    INNER JOIN component c ON p.mukey = c.mukey AND c.majcompflag = 'Yes'
    INNER JOIN chorizon ch ON c.cokey = ch.cokey AND ch.hzdept_r <= 15
    ORDER BY c.comppct_r DESC
    """
    
    payload = {
        "query": query,
        "format": "JSON+COLUMNNAME"
    }
    
    try:
        res = requests.post(url, data=payload, timeout=15)
        if res.status_code == 200:
            data = res.json()
            
            # The API returns a 'Table' array where index 0 is headers and index 1 is the data
            if "Table" in data and len(data["Table"]) > 1:
                clay, ph, om = data["Table"][1]
                
                soil_data_list.append({
                    'County': county,
                    'Soil_Clay_Pct': float(clay) if clay is not None else None,
                    'Soil_pH': float(ph) if ph is not None else None,
                    'Soil_Organic_Carbon': float(om) if om is not None else None
                })
                print(f"SUCCESS: {county} | pH: {ph} | Clay: {clay}%")
            else:
                print(f"NO DATA: {county} centroid hit an unmapped urban area or water.")
        else:
            print(f"FAILED: {county} (API Status: {res.status_code})")
            
    except Exception as e:
        print(f"ERROR: {county} - {e}")
        
    time.sleep(0.5) # Be polite to government servers

print("\nSaving True USDA Soil Data...")
df_soil = pd.DataFrame(soil_data_list)
df_soil.to_csv("TEXAS_COUNTY_SOIL_TRUE.csv", index=False)
print("Saved to 'TEXAS_COUNTY_SOIL_TRUE.csv'.")