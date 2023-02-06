import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from random import randrange
from keras import layers

class Network:
    def __init__(self, dataset):
        self.dataset = dataset
        pass

    def build_generator(self, encoding_size, num_classes):
        self.encoding_size = encoding_size
        self.generator = tf.keras.models.Sequential([
            tf.keras.layers.Dense(7 * 7 * 512, input_shape=[encoding_size+num_classes], use_bias = False),
            tf.keras.layers.Reshape((7, 7, 512)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='selu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='selu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same',activation='selu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='tanh'),
        ])

        # self.generator = tf.keras.Sequential()
        # self.generator.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=[encoding_size+num_classes]))
        # self.generator.add(layers.BatchNormalization())
        # self.generator.add(layers.LeakyReLU())
        #
        # self.generator.add(layers.Reshape((7, 7, 256)))
        # assert self.generator.output_shape == (None, 7, 7, 256)  # Note: None is the batch size
        #
        # self.generator.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        # assert self.generator.output_shape == (None, 7, 7, 128)
        # self.generator.add(layers.BatchNormalization())
        # self.generator.add(layers.LeakyReLU())
        #
        # self.generator.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        # assert self.generator.output_shape == (None, 14, 14, 64)
        # self.generator.add(layers.BatchNormalization())
        # self.generator.add(layers.LeakyReLU())
        #
        # self.generator.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        # assert self.generator.output_shape == (None, 28, 28, 1)

        self.generator.summary()
        pass

    def build_discriminator(self, num_classes):
        self.discriminator = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same',activation=tf.keras.layers.LeakyReLU(0.2), input_shape=[28, 28, 1+num_classes]),
            keras.layers.Dropout(0.4),
            keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding="same",activation=tf.keras.layers.LeakyReLU(0.2)),
            keras.layers.Dropout(0.4),
            keras.layers.Flatten(),
            keras.layers.Dense(1, activation="sigmoid")
        ])


        # self.discriminator = tf.keras.Sequential()
        # self.discriminator.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[28, 28, 1+num_classes]))
        # self.discriminator.add(layers.LeakyReLU())
        # self.discriminator.add(layers.Dropout(0.3))
        #
        # self.discriminator.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        # self.discriminator.add(layers.LeakyReLU())
        # self.discriminator.add(layers.Dropout(0.3))
        #
        # self.discriminator.add(layers.Flatten())
        # self.discriminator.add(layers.Dense(1))
        self.discriminator.summary()
        pass

    def define_loss_optimizers(self):
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
        self.loss_function = tf.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        pass

    def compile_models(self):
        self.complete_gan = tf.keras.models.Sequential([self.generator, self.discriminator])
        self.discriminator.compile(loss=self.loss_function, optimizer=self.discriminator_optimizer)
        self.discriminator.trainable = False
        self.complete_gan.compile(loss=self.loss_function, optimizer=self.generator_optimizer)
        pass

    def get_total_discrminator_loss(self, real_predictions, fake_predictions):
        # Create a real image label vector
        real_labels = tf.ones_like(real_predictions)
        # Create a fake image label vector
        fake_labels = tf.zeros_like(fake_predictions)
        # Get fake loss on fake image predictions
        fake_loss = self.loss_function(y_pred=fake_predictions, y_true = fake_labels)
        # Get real loss on real image predictions
        real_loss = self.loss_function(y_pred=real_predictions, y_true = real_labels)
        #Get total loss
        total_loss = tf.concat([real_loss, fake_loss], axis = 0)
        return total_loss

    def get_total_generator_loss(self, predictions):
        # Create a real image label vector
        trick_labels = tf.ones_like(predictions)
        # Calculate loss using the predictions and the real image label vector
        total_generator_loss = self.loss_function(y_pred=predictions, y_true=trick_labels)
        return total_generator_loss

    def train(self, epochs, num_classes):
        for epoch in range(epochs):
            for index, (real_images, one_hot_labels, one_hot_filters) in enumerate(self.dataset):
                with tf.GradientTape() as discriminator_tape:
                    # TRAIN DISCRIMINATOR
                    batch_size = real_images.shape[0]
                    # Create noise
                    noise_vector = tf.random.normal(shape=(batch_size, self.encoding_size))
                    # Concatenate the noise and one_hot_labels
                    noise_vector = tf.concat([noise_vector, one_hot_labels], axis = 1)
                    # Generate fake images from noise
                    fake_images = self.generator(noise_vector)
                    # Concatenate the fake images with the one_hot_filters
                    fake_images = tf.concat([fake_images, one_hot_filters], axis = 3)
                    # Get discriminator predictions on fake images
                    fake_predictions = self.discriminator(fake_images)
                    # Get discriminator predictions on real images
                    #Concatenate the real images with the one hot filters
                    real_images = tf.concat([real_images, one_hot_filters], axis = 3)
                    real_predictions = self.discriminator(real_images)
                    total_discriminator_loss = self.get_total_discrminator_loss(real_predictions=real_predictions, fake_predictions=fake_predictions)
                    pass

                # Calculate discriminator gradient: Differentiate total discriminator loss with respect to discriminator's trainable variables
                discriminator_gradient = discriminator_tape.gradient(total_discriminator_loss,self.discriminator.trainable_variables)
                # Use the discriminator's optimizer to apply the gradients to the discriminator's trainable variables
                self.discriminator_optimizer.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))
                with tf.GradientTape() as generator_tape:
                    # TRAIN GENERATOR
                    # Generate noise
                    noise_vector = tf.random.normal(shape=(batch_size, self.encoding_size))
                    # Concatenate the noise and one_hot_labels
                    noise_vector = tf.concat([noise_vector, one_hot_labels], axis=1)
                    # Generate fake images from noise
                    fake_images = self.generator(noise_vector)
                    # Concatenate the fake images with the one_hot_filters
                    fake_images = tf.concat([fake_images, one_hot_filters], axis = 3)
                    # Pass the images through the discriminator to get predictions
                    trick_predictions = self.discriminator(fake_images)
                    # Calculate loss using the predictions and the real image label vector
                    total_generator_loss = self.get_total_generator_loss(trick_predictions)
                    pass
                # Differentiate the loss with respect to generator's trainable variables to get the gradient
                generator_gradient = generator_tape.gradient(total_generator_loss, self.generator.trainable_variables)
                # Use the generator's optimizer to apply the gradients to the generator's trainable weights
                self.generator_optimizer.apply_gradients(zip(generator_gradient, self.generator.trainable_variables))

                if index%10 == 0:
                    print("Processed step= {0} || discriminatorloss = {1} || generator loss = {2}".format(index, tf.reduce_sum(total_discriminator_loss), tf.reduce_sum(total_generator_loss)))
                pass
            if epoch%10== 0:
                # Test the generator
                noise_vector = tf.random.normal(shape=(1, self.encoding_size))
                num = randrange(10)
                class_vector = [num]
                class_vector = tf.one_hot(class_vector, depth=num_classes)
                print("Printing {0} || one hot vector = {1}".format(num, class_vector))
                noise_vector = tf.concat([noise_vector, class_vector], axis = 1)
                fake_image = self.generator(noise_vector)
                fake_image = fake_image[0]
                plt.imshow(fake_image)
                plt.title(f"Epoch: {epoch} Number Generated: {num}")
                plt.show()
            pass
        #Save the models
        self.generator.save(f"Models/Generator{epochs}Epochs")
        self.discriminator.save(f"Models/Discriminator{epochs}Epochs")
        pass

    def train_2(self, epochs):
        generator, discriminator = self.complete_gan.layers
        for epoch in range(epochs):

            for index, (real_images, one_hot_labels, one_hot_filters) in enumerate(self.dataset):
                # print("Step number: {}".format(step + 1))
                # Get the number of images in each batch
                batch_size = real_images.shape[0]
                # Phase1 train the discriminator

                # Create noise
                noise = tf.random.normal(shape=[batch_size, self.encoding_size])
                # Use the noise to create fake images from the generator
                fake_images = generator(noise)
                # Concatenate the fake and real images into a single tensor

                discriminator_batch_input = tf.concat([fake_images, real_images], axis=0)
                # Make labels for the fake images and real images
                # For the concatenated tensor, create a label tensor for 0:fake and 1:real
                labels = tf.concat([[0 for x in range(batch_size)] + [1 for x in range(batch_size)]], axis=0)
                discriminator_batch_labels = tf.reshape(labels, shape=(batch_size * 2, 1))

                # Ensure the discriminator is training
                discriminator.set_trainable = True
                # Train the discriminator on the batch
                discriminator.train_on_batch(discriminator_batch_input, discriminator_batch_labels)

                # Phase2 train the generator

                # Create noise, just like the first time
                noise = tf.random.normal(shape=[batch_size, self.encoding_size])
                # label all generated images to be "real"
                labels = [1 for x in range(batch_size)]
                generator_labels = tf.reshape(labels, shape=(batch_size, 1))
                # Freeze the disciminator
                discriminator.set_trainable = False
                # Train the whole gan on the batch
                self.complete_gan.train_on_batch(noise, generator_labels)
                if index%10 == 0:
                    print("Processed step: {}".format(index))
                pass
            if epoch%10== 0:
                # Create some noise and make a prediction based on it
                noise = tf.random.normal(shape=[1, self.encoding_size])
                generated_image = generator(noise)
                plt.imshow(generated_image[0])
                plt.show()