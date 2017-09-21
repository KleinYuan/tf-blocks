import cv2
from models import inference

graph_fp = ''
img_fp = ''
img = cv2.imread(img_fp)
predict_net = inference.Net(graph_fp=graph_fp)
predict_net.predict(img=img)

