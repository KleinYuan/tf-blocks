import cv2


class DLVideoWriter:

    def __init__(self, name, fps=10):
        self.video_name = name
        self.fps = fps

        self.video = None
        self.height = None
        self.width = None
        self.fourcc = None

    def init_video(self, height, width):
        self.height = height
        self.width = width
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.video = cv2.VideoWriter(self.video_name, self.fourcc, self.fps, (self.width, self.height), True)

    def add_to_video(self, img):
        img = cv2.resize(img, (self.width, self.height))
        self.video.write(img)

    def finish_video(self):
        self.video.release()
