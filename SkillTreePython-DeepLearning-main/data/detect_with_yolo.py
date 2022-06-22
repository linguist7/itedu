import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import IPython

def set_model(weight_file, cfg_file):
  model = cv2.dnn.readNet(weight_file, cfg_file)
  predict_layer_names = [model.getLayerNames()[index[0] - 1] for index in model.getUnconnectedOutLayers()] 
  return model, predict_layer_names


def set_label(name_file):
  with open(name_file, 'r') as f:
    class_names = ([line.strip() for line in f.readlines()])

  class_colors = np.random.uniform(0, 255, (len(class_names), 3))
  return class_names, class_colors


def get_preds(img, model, predict_layer_names, min_confidence=0.5):
  img_h, img_w, img_c = img.shape
  blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
  model.setInput(blob)

  boxes = []
  confidences=[]
  class_ids = []
  preds_each_layers = model.forward(predict_layer_names)
  for preds in preds_each_layers:
    for pred in preds:
      box, confidence, class_id = pred[:4], pred[4], np.argmax(pred[5:])
      if confidence > min_confidence:
        x_center, y_center, w, h = box
        x_center, w = int(x_center*img_w), int(w*img_w)
        y_center, h = int(y_center*img_h), int(h*img_h)
        x, y = x_center-int(w/2), y_center-int(h/2)
        
        boxes.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)
  return boxes, confidences, class_ids


def draw_result(img, 
                boxes, confidences, class_ids,
                class_names, class_colors,
                min_confidence=.5,
                font_size=.6):
  selected_box_idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

  carnum_model = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_russian_plate_number.xml')
  carnum_model.load(cv2.samples.findFile(cv2.data.haarcascades +'haarcascade_russian_plate_number.xml')) 

  for bi, (x, y, w, h) in enumerate(boxes):
    if bi in selected_box_idx:            
      class_id = class_ids[bi]
      color = class_colors[class_id]
      class_name = class_names[class_id]

      cv2.rectangle(img, (x, y), (x+w, y+h), color , 2)
      cv2.putText(img, 'class_name', (x, y), cv2.FONT_ITALIC, 0.5, color, 2)
      
      car = img[y:y+h, x:x+w]
      carnums_pred = carnum_model.detectMultiScale(car)
      for (x2, y2, w2, h2) in carnums_pred:                           
        cv2.rectangle(car, (x2, y2), (x2+w2, y2+h2), (0,0,255) , 2)
  IPython.display.clear_output(wait=True)
  cv2_imshow(img)
    

def img2detect(img, model, predict_layer_names, class_names, class_colors,
               min_confidence=.5,
               font_size=.6):
    boxes, confidences, class_ids = get_preds(img, model, predict_layer_names, min_confidence=0.5)
    draw_result(img,
                boxes, confidences, class_ids,
                class_names, class_colors,
                min_confidence=min_confidence,
                font_size=font_size)
