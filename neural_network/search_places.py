import overpy

import requests


def geocode_address_nominatim(address):
    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": address,
        "format": "json"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data:
        latitude = data[0]["lat"]
        longitude = data[0]["lon"]
        return f"{latitude},{longitude}"
    else:
        return "Не удалось найти координаты для указанного адреса."


def tourism(address):
    cord = geocode_address_nominatim(address)

    api = overpy.Overpass()

    arr_ans = []


    result = api.query(f"""
        node["tourism"="museum"](around:1000,{cord});
        out;
    """)

    for node in result.nodes:
        arr_ans.append(node.tags.get("name", "неизвестно"))
        # print("Музей:", node.tags.get("name", "неизвестно"))

    # Поиск парков
    result = api.query(f"""
        node["leisure"="park"](around:1000,{cord});
        out;
    """)

    for node in result.nodes:
        arr_ans.append(node.tags.get("name", "неизвестно"))
        # print("Парк:", node.tags.get("name", "неизвестно"))

    # Поиск кинотеатров
    result = api.query(f"""
        node["amenity"="cinema"](around:1000,{cord});
        out;
    """)

    for node in result.nodes:
        arr_ans.append(node.tags.get("name", "неизвестно"))
        # print("Кинотеатр:", node.tags.get("name", "неизвестно"))

    result = api.query(f"""
        node["amenity"="gallery"](around:1000,{cord});
        out;
    """)

    for node in result.nodes:
        arr_ans.append(node.tags.get("name", "неизвестно"))
        # print("gallery:", node.tags.get("name", "неизвестно"))


    return arr_ans


def working(address):
    cord = geocode_address_nominatim(address)
    api = overpy.Overpass()
    arr_ans = []


    result = api.query(f"""
        node["amenity"="cafe"](around:1000,{cord});
        out;
    """)

    for node in result.nodes:
        arr_ans.append(node.tags.get("name", "неизвестно"))
        # print("Cafe:", node.tags.get("name", "неизвестно"))

    # Поиск парков
    result = api.query(f"""
        node["amenity"="restaurant"](around:1000,{cord});
        out;
    """)

    for node in result.nodes:
        arr_ans.append(node.tags.get("name", "неизвестно"))
        # print("restaurant:", node.tags.get("name", "неизвестно"))

    return arr_ans



if __name__ == '__main__':
    print(working("Москва, проспект Мира 150"))
