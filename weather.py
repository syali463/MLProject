import requests
API_KEY="20F76B03-6A03-3D73-BFA1-C8C98B794FB8"
#response = requests.get("https://power.larc.nasa.gov/api/temporal/monthly/regional?start=2005&end=2025&latitude-min=29.65&latitude-max=33.55&longitude-min=-102.98&longitude-max=-94.24&community=ag&parameters=T2M_MIN&format=csv&units=metric&header=true")
cropResponse = requests.get(f"https://quickstats.nass.usda.gov/api/api_GET/?key=${API_KEY}&commodity_desc=CORN&year__GE=2010")
if cropResponse.status_code == 200:
    # with open('T2M_MIN.csv', 'wb') as file:
    #     file.write(response.content)
    print(cropResponse.headers)
else:
    print(f"Failed to fetch data. Status: {cropResponse.reason}")