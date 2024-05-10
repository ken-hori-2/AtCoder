# import googlemaps

# import os
# from dotenv import load_dotenv
# # .envファイルの内容を読み込見込む
# load_dotenv()

# MAPS_API_KEY = os.environ['GOOGLE_MAPS_API_KEY']

# def geocode(address):
#    try:
#        gmaps = googlemaps.Client(key=MAPS_API_KEY)
#        result = gmaps.geocode(address)
#        lat = result[0]['geometry']['location']['lat']
#        lng = result[0]['geometry']['location']['lng']

#        return lat, lng
#    except:
#        return None, None

# test1, test2 = geocode("玉川学園前駅")
# print(test1, test2)

import requests

endpoint = "https://ipinfo.io"

headers= {
    
}
params={
}

result = requests.get(endpoint, headers=headers, params=params)

res = result.json()

print(res)

