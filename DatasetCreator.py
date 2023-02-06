from keras.datasets import mnist
import tensorflow as tf

class DatasetCreator:
    def get_datasets(self, batch_size):
        (data, labels), (_, _) = mnist.load_data()
        data = tf.cast(data, dtype=tf.float32)
        mnist_shape = data.shape
        num_classes = 10
        #Images
        data = tf.expand_dims(data, axis=-1)
        #One hot labels
        one_hot_labels = tf.one_hot(labels, depth=num_classes)
        #One hot filters
        one_hot_filters = tf.expand_dims(tf.expand_dims(one_hot_labels, axis = -1), axis = -1)
        one_hot_filters = tf.reshape(one_hot_filters, shape=(mnist_shape[0], 1,1,num_classes))
        one_hot_filters = tf.tile(one_hot_filters, [1,28,28, 1])

        dataset = tf.data.Dataset.from_tensor_slices((data, one_hot_labels, one_hot_filters))
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
        dataset = dataset.shuffle(buffer_size=batch_size+1)
        dataset = dataset.map(self.normalize_map)
        return dataset

    def normalize_map(self, datapoint, label, one_hot_filter):
        datapoint = datapoint / 255.
        return datapoint, label, one_hot_filter
