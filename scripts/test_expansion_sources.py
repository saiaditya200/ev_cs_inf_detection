import requests
import json

session = requests.Session()
session.headers.update({
    "User-Agent": "EVChargingResearch/1.0 (Academic GIS project; contact: student@univ.edu)",
    "Accept": "application/json",
})

bb = "15.8,77.2,19.9,81.8"
query = f"""[out:json][timeout:90];
(
  node["amenity"="charging_station"]({bb});
  way["amenity"="charging_station"]({bb});
  node["fuel:electricity"="yes"]({bb});
  node["charging_station"="yes"]({bb});
);
out body center;"""

print("Sending Overpass query with proper User-Agent...")
resp = session.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=90)
print("Status:", resp.status_code)
data = resp.json()
elements = data.get("elements", [])
print(f"Total Overpass elements: {len(elements)}")
for i, e in enumerate(elements[:10]):
    tags = e.get("tags", {})
    lat = e.get("lat") or e.get("center", {}).get("lat")
    lon = e.get("lon") or e.get("center", {}).get("lon")
    print(f" {i+1}. ID:{e.get('id')} Name:{tags.get('name')} Op:{tags.get('operator')} Coord:({lat}, {lon})")
