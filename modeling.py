from networks import InceptionNet
from data_formatting import build_tensorflow_dataset

data_set = build_tensorflow_dataset()
network = InceptionNet()

network.train(data_set, 1000)
