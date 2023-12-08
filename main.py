# -*- coding: utf-8 -*-

import neural_network.description_generation as dg
import neural_network.furniture_recognition as fr
import neural_network.search_places as sp


from flask import Flask, render_template, request, send_file, make_response, jsonify
from werkzeug.utils import secure_filename
import os
from time import sleep
import random

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'neural_network/image'

# допустимые расширения файлов
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


data_generation = {}
k = 0


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template("index.html")

@app.route("/<arg1>")
def arg1(arg1):
    arg1 = int(arg1)

    if data_generation[arg1][-1]:
        sleep(10)
        return ""


    data_generation[arg1][-1] = True

    img_s = "В номерах присудствуют элементы мебели, такие как - "

    for i in data_generation[arg1][5]:
        img_s = fr.furniture_recognition(f"neural_network/image/{i}")

    text = "перепиши другими словами, выделяя все самое важное"

    if data_generation[arg1][1] == "1":
        text += " используя художественный стиль,"
    if data_generation[arg1][1] == "2":
        text += " используя деловой стиль,"

    if data_generation[arg1][2]:
        text += f' и подробнее рассписывая о "{data_generation[arg1][2]}",'
    if data_generation[arg1][3]:
        text += f' удали все что связанно со словом "{data_generation[arg1][3]},"'

    if data_generation[arg1][0] == "1":
        text += " размером до 50 слов"
    elif data_generation[arg1][0] == "2":
        text += " размером до 100 слов"
    else:
        text += " размером до 200 слов"

    text += ". Примени все это к данному описанию: \n" + data_generation[arg1][4]

    print(text)

    with open('templates/updata.html', 'r', encoding="utf-8") as file:
        data = file.read()

    data = data.replace('<-old->', data_generation[arg1][4])
    data = data.replace('<-new->', dg.generate_text(text + " " + data_generation[arg1][4]))
    # data = data.replace('<-new->', 'Сеть отелей "Космос" - это современный отель недалеко от Садового кольца, который предлагает своим гостям комфортное размещение в номерах разных категорий. Номера оформлены в едином лаконичном стиле, а кровати - с ортопедическими матрасами. В каждом номере есть Wi-Fi, кабельное телевидение, система управления освещением и кондиционер. Также доступны номера для людей с ограниченными возможностями. В отеле есть конференц-залы разной вместимости, чтобы можно было с комфортом организовать мероприятие.')

    ans = "Рядом с отелем есть следующее туристические места:"
    place = sp.tourism(data_generation[arg1][6])

    if len(place) > 5:
        data = data.replace("<-tour->", ans + "\n·" + "\n·".join(random.sample(place, 5)))
    elif len(place) == 0:
        data = data.replace("<-tour->", "В окрестностях отеля нет достопримечательностей")
    else:
        data = data.replace("<-tour->", ans + "\n·" + "\n·".join(place))

    ans = "Рядом с отелем есть места для деловых встреч:"
    place = sp.working(data_generation[arg1][6])

    if len(place) > 5:
        data = data.replace("<-work->", ans + "\n·" + "\n·".join(random.sample(place, 5)))
    elif len(place) == 0:
        data = data.replace("<-work->", "В окрестностях отеля нет деловых мест")
    else:
        data = data.replace("<-work->", ans + "\n·" + "\n·".join(place))

    data = data.replace("<-weather->", "Погода на выходных следующая:\n·Суббота: Облачно, температура -11С - рекомендуем сходить в закрытое помещение, например 'Музей истории ВДНХ'\n·Воскресенье: Облачно, температура -9С - рекомендуем сходить в закрытое помещение, например 'Царь-макет'")

    sleep(5)
    print("ok")
    return data


@app.route('/api', methods=['POST'])
def process_data():
    data = request.form  # Получаем текстовые данные из запроса
    file = request.files['file']  # Получаем файл из запроса
    print(data)
    #
    # if file and allowed_file(file.filename):
    #     filename = secure_filename(file.filename)
    #     file.save(os.path.join("neural_network/image", filename))

    text = "Отредактируй данный текст выделяя важное"

    if data["style"] == "1":
        text += " используя художественный стиль"
    if data["style"] == "2":
        text += " используя деловой стиль"

    if data["add_word"]:
        text += f' и обязательно используя слово "{data["add_word"]}"'
    if data["stop_word"]:
        text += f' и не используя слова "{data["stop_word"]}"'

    if data["size"] == "1":
        text += " размером до 50 слов"
    elif data["size"] == "2":
        text += " размером до 100 слов"
    else:
        text += " размером до 200 слов"

    text += ": \n" + data["text"]


    return jsonify({"text": dg.generate_text(text)})



@app.route('/upload', methods=['POST'])
def upload_file():
    global k

    if 'file' not in request.files:
        return 'No file part'

    print(request.form)
    file = request.files['file']

    files = []

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join("neural_network/image", filename))
        files.append(filename)

    k = k + 1
    data_generation[k] = [request.form.get("size"), request.form.get("style"), request.form["text_im"], request.form["text_notim"], request.form["text"], files, request.form["address"], False]

    with open('templates/generation.html', 'r', encoding="utf-8") as file:
        data = file.read()

    data = data.replace('<-id->', str(k))

    with open(f'templates/generation{k}.html', 'w', encoding="utf-8") as new_file:
        new_file.write(data)

    return render_template(f"generation{k}.html")


if __name__ == '__main__':
    app.run(debug=True)
