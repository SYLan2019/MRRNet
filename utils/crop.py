import cv2
import argparse
import os
import time

"""
Crop.py use for crop face area and it's parsing map in image.
"""
parser = argparse.ArgumentParser(description="Crop out the face in image and parsing map")
parser.add_argument('-input_dir', type=str, default="input", help="input image directory")
parser.add_argument('-lable_dir', type=str, default="label", help="parsing map directory")
parser.add_argument('-fcod', type=str, default="result", help="face crop output dir")
parser.add_argument('-flcod', type=str, default="result", help="face lable (parsing map) crop output dir")
parser.add_argument('-size', type=int, default=128, help="size of resize function")
args = parser.parse_args()

face_cascade = cv2.CascadeClassifier("haar_cascade.xml")
num = 0
for img in os.listdir(args.input_dir):
    img_path = os.path.join(args.input_dir, img)
    in_img = cv2.imread(img_path)
    gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    lable_name = img.split(".")[0]
    lable_path = os.path.join(args.lable_dir, lable_name, lable_name + "_lbl0:q1.png")

    lable_img = cv2.imread(lable_path)
    i = 0
    top = 100
    down = 80
    for (x, y, w, h) in faces:
        while y - top < 0:
            top = top - 5
        while y + h + down > in_img.shape[0]:
            down = down - 5
        face = in_img[y - top:y + h + down, x:x + w]
        face = cv2.resize(face, dsize=(args.size, args.size))

        parsing_map = lable_img[y - top:y + h + down, x:x + w]
        parsing_map = cv2.resize(parsing_map, dsize=(args.size, args.size))
        cv2.imwrite(args.fcod + str(i) + "_" + img, face)
        cv2.imwrite(args.flcod + str(i) + "_" + img, parsing_map)
        i = i + 1
        top = 100
        down = 80
    print("\rprocess " + img + " success," + str(num), end="")
    num = num + 1
print("finish.")
