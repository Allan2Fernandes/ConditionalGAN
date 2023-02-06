import Network
import tensorflow as tf
import numpy as np
import DatasetCreator
import matplotlib.pyplot as plt



batch_size = 256
num_classes = 10
dataset = DatasetCreator.DatasetCreator().get_datasets(batch_size)

network = Network.Network(dataset)
network.build_generator(encoding_size=32, num_classes=num_classes)
network.build_discriminator(num_classes=num_classes)
network.define_loss_optimizers()
#network.compile_models()
network.train(100, num_classes=num_classes)







