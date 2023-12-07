# -*- coding: utf-8 -*-

import neural_network.description_generation as dg
import neural_network.furniture_recognition as fr


from flask import Flask, render_template, request, send_file, make_response, jsonify
from werkzeug.utils import secure_filename
import os
from time import sleep

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'neural_network/image'

# допустимые расширения файлов
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


data_generation = {}
k = 0

# Функция для проверки допустимых расширений файлов
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template("index.html")

@app.route("/<arg1>")
def arg1(arg1):
    arg1 = int(arg1)

    if data_generation[arg1][-1]:
        sleep(300)
        return ""


    data_generation[arg1][-1] = True

    img_s = "В номерах присудствуют элементы мебели, такие как - "

    # for i in data_generation[arg1][5]:
    #     img_s = fr.furniture_recognition(f"neural_network/image/{i}")

    text = "перепиши другими словами, выделяя важное"

    if data_generation[arg1][2]:
        text += f'и обязательно используя слово "{data_generation[arg1][4]}"'
    if data_generation[arg1][3]:
        text += f'и не используя слова "{data_generation[arg1][3]}"'

    text += ":"


    with open('templates/updata.html', 'r', encoding="utf-8") as file:
        data = file.read()

    data = data.replace('<-old->', data_generation[arg1][4])
    data = data.replace('<-new->', dg.generate_text(text + " " + data_generation[arg1][4]))
    print("ok")
    return data


@app.route('/api', methods=['POST'])
def process_data():
    data = request.form  # Получаем текстовые данные из запроса
    file = request.files['file']  # Получаем файл из запроса
    print(data)
    # Проверяем, что файл соответствует разрешенным типам
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join("neural_network/image", filename))

        # Теперь у тебя есть доступ к файлу, и ты можешь обработать его, как тебе нужно

        # Возвращаем результат обработки в формате JSON
        return jsonify({'processed_text': data.get('text', ''), 'file_uploaded': True})
    else:
        # Если файл не был загружен или имеет недопустимый формат, возвращаем ошибку
        return jsonify({'error': 'Invalid file or no file provided.'}), 400


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
    data_generation[k] = [request.form.get("size"), request.form.get("style"), request.form["text_im"], request.form["text_notim"], request.form["text"], files, False]

    with open('templates/generation.html', 'r', encoding="utf-8") as file:
        data = file.read()

    data = data.replace('<-id->', str(k))

    with open(f'templates/generation{k}.html', 'w', encoding="utf-8") as new_file:
        new_file.write(data)

    return render_template(f"generation{k}.html")


if __name__ == '__main__':
    app.run(debug=True)
