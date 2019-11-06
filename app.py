import numpy as np
import cv2
import textwrap

from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, redirect, url_for
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__, static_url_path='', static_folder=dir_path + '/static/')


def extract_classes():
    with open(dir_path + '/yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def extract_needed_classes():
    with open(dir_path + '/my_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def detect_objects(frame, outs, needed_classes_ids):
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    width = frame.shape[1]
    height = frame.shape[0]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2 and class_id in needed_classes_ids:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    return boxes, class_ids, conf_threshold, confidences, nms_threshold


def draw_prediction(img, classes, class_id, needed_classes_ids, colors, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = colors[needed_classes_ids.index(class_id)]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def frame_object_detection(frame, all_classes, needed_classes_ids, colors):
    yolov_weights_path = dir_path + '/yolov3.weights'
    yolov_config_path = dir_path + '/yolov3.cfg'
    scale = 0.00392

    net = cv2.dnn.readNet(yolov_weights_path, yolov_config_path)
    blob = cv2.dnn.blobFromImage(frame, scalefactor=scale, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    boxes, class_ids, conf_threshold, confidences, nms_threshold = detect_objects(frame, outs, needed_classes_ids)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(frame, all_classes, class_ids[i], needed_classes_ids, colors, round(x), round(y), round(x + w),
                        round(y + h))
    if len(class_ids) > 0:
        return True
    else:
        return False


def draw_appeared_or_disappeared(frame, val):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    if val:
        text = 'Появился'
    else:
        text = 'Пропал'
    image_width, image_height = image.size
    y_text = image_height - 100
    fontsize = 40
    lines = textwrap.wrap(text, width=40)
    font = ImageFont.truetype(dir_path + "/Arial.ttf", fontsize)
    for line in lines:
        line_width, line_height = font.getsize(line)
        draw.text(((image_width - line_width) / 2, y_text),
                  line, font=font)
        y_text += line_height
    return np.array(image)


def make_video():
    cap = cv2.VideoCapture(dir_path + '/Computer Vision Test Video.ogv')
    out1 = cv2.VideoWriter(dir_path + '/static/output.webm', cv2.VideoWriter_fourcc(*'vp80'), int(cap.get(5)),
                           (int(cap.get(3)), int(cap.get(4))))
    cup_in_out = [False] * 18
    all_classes = extract_classes()
    needed_classes = extract_needed_classes()
    needed_classes_ids = [all_classes.index(i) for i in needed_classes]
    colors = np.random.uniform(0, 255, size=(len(needed_classes), 3))
    in_out = False
    while cap.isOpened():
        ret, frame = cap.read()
        try:
            cup_in_out.append(frame_object_detection(frame, all_classes, needed_classes_ids, colors))
            cup_in_out.pop(0)
            for i in cup_in_out:
                if i:
                    in_out = True
                    break
                in_out = False
            frame = draw_appeared_or_disappeared(frame, in_out)
            out1.write(frame)
        except:
            pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out1.release()

@app.route('/make_video')
def main():
    make_video()
    return redirect(url_for('show_video'), code=302)


@app.route('/')
def show_video():
    return render_template("index.html")


if __name__ == '__main__':
    # app.run()
    app.run(host='0.0.0.0', port='80')
