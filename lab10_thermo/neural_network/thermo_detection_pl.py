from os.path import join
import cv2
import numpy as np

############################### Funkcje #################################

def detect(net, img):
    size = img.shape
    height = size[0]
    width = size[1]
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    boxes = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
    return boxes

def filter_boxes(boxes):
    all_paired_boxes = list()
    for ii, box1 in enumerate(boxes):
        x1, y1, w1, h1 = box1
        center_x1 = x1 + int(w1 / 2)
        center_y1 = y1 + int(h1 / 2)
        to_connect = [ii]
        for jj, box2 in enumerate(boxes):
            if jj != ii:
                x2, y2, w2, h2 = box2
                center_x2 = x2 + int(w2 / 2)
                center_y2 = y2 + int(h2 / 2)
                if abs(center_x2 - center_x1) < 10 and abs(center_y2 - center_y1) < 10:
                    to_connect.append(jj)
        all_paired_boxes.append(to_connect)
    all_paired_boxes = sorted(all_paired_boxes, key=lambda x: len(x), reverse=True)
    all_paired = list()
    final_boxes = list()
    for conn in all_paired_boxes:
        if all([a not in all_paired for a in conn]):
            for a in conn:
                all_paired.append(a)
            final_boxes.append(conn)
    out_boxes = [[int(sum([boxes[i][a] for i in elem]) / len(elem)) for a in range(4)] for elem in final_boxes]
    return out_boxes

def IoU(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    left = max([x1, x2])
    right = min([x1 + w1, x2 + w2])
    top = max([y1, y2])
    bottom = min([y1 + h1, y2 + h2])
    area1 = max([(right - left), 0]) * max([(bottom - top), 0])
    area2 = (w1 * h1) + (w2 * h2) - area1
    if area2 == 0:
        return 0
    return area1 / area2

############# METHOD ###############
# FUSION = "EARLY"  # or 
FUSION = "LATE"

############# Ścieżki ##############
test_rgb = "test_rgb"
test_thermal = "test_thermal"
###################################

net_fus = None
net_therm = None
net_rgb = None
if FUSION == "EARLY":
    net_fus = cv2.dnn.readNet('yolov3_training_last_f.weights', 'yolov3_testing_f.cfg')
if FUSION == "LATE":
    net_therm = cv2.dnn.readNet('yolov3_training_last_t.weights', 'yolov3_testing_t.cfg')
    net_rgb = cv2.dnn.readNet('yolov3_training_last_c.weights', 'yolov3_testing_c.cfg')

for i in range(200, 300):  # Można zwiększyć do 518
    path_rgb = join(test_rgb, f"img{i}.png")
    path_thermal = join(test_thermal, f"img{i}.png")
    img_rgb = cv2.imread(path_rgb)
    img_thermal = cv2.imread(path_thermal)
    img_thermal = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2GRAY)
    out_img = None
    boxes = None

    if FUSION == "EARLY":
        # TODO1 rozwiązany:
        new_fus = np.zeros_like(img_rgb)
        new_fus[:, :, :2] = img_rgb[:, :, :2]
        new_fus[:, :, 2] = np.maximum(img_rgb[:, :, 2], img_thermal)
        new_fus = new_fus.astype("uint8")

        out_img = new_fus.copy()
        boxes = detect(net_fus, new_fus)

    if FUSION == "LATE":
        out_img = img_rgb.copy()
        Rect1 = detect(net_therm, img_thermal)
        Rect2 = detect(net_rgb, img_rgb)

        # TODO2 rozwiązany:
        boxes_iou = []
        for idx1, r1 in enumerate(Rect1):
            for idx2, r2 in enumerate(Rect2):
                iou_val = IoU(r1, r2)
                if iou_val > 0:
                    boxes_iou.append([(idx1, idx2), iou_val])

        boxes_iou = sorted(boxes_iou, key=lambda a: a[1], reverse=True)

        Rect1_paired = []
        Rect2_paired = []
        paired_boxes = []

        for pair, _ in boxes_iou:
            idx1, idx2 = pair
            if idx1 not in Rect1_paired and idx2 not in Rect2_paired:
                paired_boxes.append((idx1, idx2))
                Rect1_paired.append(idx1)
                Rect2_paired.append(idx2)

        boxes = []
        for idx1, idx2 in paired_boxes:
            r1 = Rect1[idx1]
            r2 = Rect2[idx2]
            avg_r = [int((r1[j] + r2[j]) / 2) for j in range(4)]
            boxes.append(avg_r)

    out_boxes = filter_boxes(boxes)
    for box in out_boxes:
        x, y, w, h = box
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.imshow('Image', out_img)
    cv2.waitKey(10)

cv2.destroyAllWindows()
