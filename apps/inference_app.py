from models import inference


class InferenceApp:

    def __init__(self, graph_fp):
        self.graph_fp = graph_fp
        self.predictor = None
        self.prediction = None

    def _init_graph(self):
        self.predictor = inference.Net(graph_fp=self.graph_fp)

    def predict(self, img):

        self.predictor.predict(img=img)
        self.prediction = self.predictor.get_prediction()

