import requests
import json

session = requests.Session()

# 1. 메인 페이지 먼저 방문해서 쿠키 획득
session.get(
    "https://zippoom.com/",
    headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
)

# 2. API 호출
res = session.get(
    "https://live.zippo-om.com/api/v2/map/price",
    params={
        "minLat": 37.589520666666665,
        "minLon": 127.05454726321395,
        "maxLat": 37.59510313333333,
        "maxLon": 127.0587172088093,
        "tradeTypes": "전세,월세",
        "residenceTypes": "APARTMENT,OFFICETEL,HOUSE,ETC",
        "maemaeMin": "", "maemaeMax": "",
        "jeonseMin": "", "jeonseMax": "",
        "depositMin": "", "depositMax": "",
        "monthlyFeeMin": "", "monthlyFeeMax": "",
        "ratingFrom": "", "ratingTo": "",
        "minPy": "", "maxPy": "",
    },
    headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "ko-KR,ko;q=0.9",
        "Referer": "https://zippoom.com/",
        "Origin": "https://zippoom.com",
    }
)

# 3. JSON 파일로 저장
data = res.json()
with open("zippoom_result.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("저장 완료! zippoom_result.json 확인하세요")
print(json.dumps(data, ensure_ascii=False, indent=2))