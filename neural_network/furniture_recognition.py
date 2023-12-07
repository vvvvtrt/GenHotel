from imageai.Detection import ObjectDetection
import os


def furniture_recognition(file_name):
    important_furniture = {"diningtable": "стол на котором можно работать",
                           "pottedplant": "цветы в горшке для создания уютной обстановки",
                           "microwave": "микроволновка",
                           "toaster": "тостер",
                           "hairdryer": "фен",
                           "tvmonitor": "телевизор для хорошего времяпрепровождения",
                           "refrigerator": "холодильник"}

    important_furnitures = {"diningtable": "столов на котором можно работать",
                            "pottedplant": "цветы в горшке для создания уютной обстановки",
                            "microwave": "микроволновок",
                            "toaster": "тостеров",
                            "hairdryer": "фенов",
                            "tvmonitor": "телевизоров для хорошего времяпрепровождения",
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

    answer_text = ""

    for i in dict_in:
        if dict_in[i] == 1:
            answer_text += important_furniture[i] + ","
        else:
            answer_text += "несколько " + important_furnitures[i] + ", "

    return answer_text


if __name__ == '__main__':
    print(furniture_recognition("image/test1.jpeg"))
