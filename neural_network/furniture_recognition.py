from imageai.Detection import ObjectDetection
import os


def furniture_recognition(file_name):
    important_furniture = {"diningtable": "стол",
                           "pottedplant": "цветы в горшке",
                           "sofa": "диван",
                           "chair": "стул",
                           "microwave": "микроволновка",
                           "toaster": "refrigerator",
                           "hairdryer": "фен",
                           "tvmonitor": "телевизор",
                           "oven": "печь",
                           "refrigerator": "холодильник"}

    important_furnitures = {"diningtable": "столов",
                            "pottedplant": "цвет в горшке",
                            "sofa": "диванов",
                            "chair": "стульев",
                            "microwave": "микроволновок",
                            "toaster": "тостеров",
                            "hairdryer": "фенов",
                            "tvmonitor": "телевизоров",
                            "oven": "печей",
                            "refrigerator": "холодильников"}

    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path, "yolov3.pt"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, file_name),
                                                 output_image_path=os.path.join(execution_path, "temp.jpg"),
                                                 minimum_percentage_probability=30)

    dict_in = {}

    for eachObject in detections:
        if eachObject["name"] in dict_in:
            dict_in[eachObject["name"]] += 1
        elif eachObject["name"] in important_furniture:
            dict_in[eachObject["name"]] = 1

        # print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        # print("--------------------------------")

    answer_text = "В номерах присудствует следующая мебель: "

    for i in dict_in:
        if dict_in[i] == 1:
            answer_text += important_furniture[i] + ","
        else:
            answer_text += "несколько " + important_furnitures[i] + ", "

    return answer_text


if __name__ == '__main__':
    print(furniture_recognition("test3.jpeg"))
