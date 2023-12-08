import requests

# Функция для получения данных о погоде
def get_weather_data(start_date, end_date):
    api_key = "7516b292fae6615b8d20686e1f83933e"  # Замените на свой API ключ
    base_url = "https://api.weather.com/data/2.5/forecast"

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "apikey": api_key
    }

    response = requests.get(base_url, params=params)
    data = response.json()
    return data

# Функция для определения характера погоды
def determine_weather_condition(weather_data):
    sunny_days = []
    cloudy_days = []

    for day in weather_data:
        if day["weather"] == "sunny":
            sunny_days.append(day["date"])
        else:
            cloudy_days.append(day["date"])

    return sunny_days, cloudy_days

# Основная часть программы
start_date = "2023-12-08"
end_date = "2023-12-31"

data = get_weather_data(start_date, end_date)
sunny_days, cloudy_days = determine_weather_condition(data)

print("Солнечные дни:")
for day in sunny_days:
    print(day, "- пойти в парк")

print("Пасмурные дни:")
for day in cloudy_days:
    print(day, "- пойти в музей")
