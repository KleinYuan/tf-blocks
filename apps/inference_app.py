import cv2
import numpy as np
from models import inference, video


# TODO: Before you run this, make sure you understand what is going on below and customzie it!
# TODO: Don't just run it.

# Predict one img
graph_fp = ''
img_fp = ''
img = cv2.imread(img_fp)
predict_net = inference.Net(graph_fp=graph_fp)
predict_net.predict(img=img)
prediction = predict_net.get_prediction()
prediction_str = np.array2string(prediction, precision=2, separator=',',
                                 suppress_small=True)

# save imgs into
# img_fp = '../data/generic_data/imgs/'
# save_fp = '%svideo.mp4' % img_fp
# video_writer = video.DLVideoWriter(name=save_fp)
# video_writer.init_video(height=100, width=100)
#
# for i in range(1, 20):
#     fp = img_fp + '%s.jpg' % i
#     print fp
#     img = cv2.imread(fp)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     font_color = (0, 255, 0)
#     line_type = 2
#     offset = 20
#     cv2.putText(img, prediction_str,
#                 (offset, offset),
#                 font,
#                 font_scale,
#                 font_color,
#                 line_type)
#
#     video_writer.add_to_video(img=img)
# video_writer.finish_video()

