from teachable_machine import TeachableMachine
import cv2 as cv

tm_model_path = "model_files/keras_model.h5"
tm_labels_path = "model_files/labels.txt"

cap = cv.VideoCapture(0)

model = TeachableMachine(model_path=tm_model_path,
                         labels_file_path=tm_labels_path)

image_path = "screenshot.jpg"

while True:
    _, img = cap.read()
    cv.imwrite(image_path, img)

    result = model.classify_image(image_path)
    
    if "Mask" in str(result["class_name"]):
        print(result["class_index"])
        print("Safe, it contains a mask.")
    elif "Without" in str(result["class_name"]):
        print(result["class_index"])
        print("Danger, this person isn't wearing a mask.")

    print("Confidence:", result["class_confidence"])

    cv.imshow("Video Stream", img)

    k = cv.waitKey(1)

    if k % 255 == 27: # close video stream when Esc is pressed
        break
