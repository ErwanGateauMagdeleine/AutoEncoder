import os
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, \
                                    Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np


class AutoEncoder:

    """
    AutoEncoder represents a deep convolutional auto encoder architecture with
    mirrored encoder and decoder components.
    """

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        """ Initialize an instance of AutoEncoder class.

            Arguments:
                input_shape: A shape tuple (integer), not including the batch size representing
                             the input data shape. For instance input_shape=(32,) indicates that
                             the expected input will be batches of 32-dimentional vectors.
                conv_filters: List (integer) representing the dimentionality of the output space
                              of the convolutional layers (i.e. the number of output filter ).
                              For instance conv_filter=[32,64] indicates that the model is composed
                              of 2 convolutional layers whose respectly have 32 and 64 output filters.
            conv_kernels: List (tuple) specifying the height and width of the 2D convolution
                          window for each convolutional layer.
            conv_strides: List (tuple of 2 integers) specifying the strides of the convolution along the
                          height and width.
            latent_space_dim: A shape tuple (integer), not including the batch size representing the size
                              of the latent space.
        """
        assert len(conv_filters) == len(conv_kernels) == len(conv_strides), \
        ("len(conv_filters),  len(conv_kernels) and len(conv_strides) not matching")

        self.input_shape = input_shape
        self.conv_filters  = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None
    
        self._build()

    def summary(self):
        """ Print the summary of the model. """
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate):
        """ Compiles the model. """
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train(self, x_train, batch_size, num_epochs):
        """ Train the model on the given input data
        """
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)

    def save(self, save_folder="."):
        """ Save the trained model. """
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weigths(save_folder)

    def _create_folder_if_it_doesnt_exist(self, folder):
        """ Creates the folder in which the model will be saved if
            it doesn't exists already. """
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        """ Save the parameters of the model. """
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weigths(self, save_folder):
        """Save the weigths of the model. """
        save_path = os.path.join(save_folder, "weigths.h5")
        self.model.save_weights(save_path)

    @classmethod
    def load(cls, save_folder="."):
        """ Loads the model. """
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = AutoEncoder(*parameters)

        weigths_path =os.path.join(save_folder, "weigths.h5")
        autoencoder.load_weigths(weigths_path)

        return autoencoder

    def load_weigths(self, weights_path):
        """ Loads the weigths of the model. """
        self.model.load_weights(weights_path)

    def _build(self):
        """ Build the complete model. """
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        """ Build the full AutoEncoder """
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        """ Build the Decoder. """
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")
    
    def _add_decoder_input(self):
        """ Creates the input layer of decoder. 
        
            Returns:
                A tensor.
        """
        return Input(self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        """ Add a dense layer.

            Arguments:
                decoder_input: A tensor.
            Returns:
                A tensor
        """
        num_neurons = np.prod(self._shape_before_bottleneck)  #[1, 2, 4] -> 8
        return Dense(num_neurons, name="decoder_dense")(decoder_input)

    def _add_reshape_layer(self, dense_layer):
        """ Transform the flatten shaped tensor to a tensor with shape_before_bottleneck
            shape.

            Arguments:
                dense_layer: A tensor.
            Returns:
                A tensor
        """
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """ Add convolutional transpose blocks. Loop through all the conv layers in reverse order
            and stop at the first layer.

            Arguments:
                x: A tensor.
            Returns:
                A tensor
        """
        for layer_index in reversed(range(1, self._num_conv_layers)): 
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        """ Adds a transpose convolutional block to a graph of layers. consisting of conv transpose 2d
            + ReLU + batch normalization.

            Arguments:
                encoder_input: A tensor.
            Returns:
                A tensor
        """
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )

        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)

        return x

    def _add_decoder_output(self, x):
        """ Add the output layer of the encoder.
        """
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_encoder(self):
        """ Build the encoder. """
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        """ Creates the input layer of encoder. 
        
            Returns:
                A tensor.
        """
        return Input(self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        """ Creates all convolutional blocks in encoder.
            Arguments:
                encoder_input: A tensor.
            Returns:
                A tensor
        """
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """ Adds a convolutional block to a graph of layers, consisting of conv 2d + ReLU + batch normalization.

            Arguments:
                layer_index: Integer representing the layer index.
                x: A tensor.
            Returns:
                A tensor
        """
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_index}"
        )
        x = conv_layer(x)
        x = ReLU()(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_index}")(x)
        return x
    
    def _add_bottleneck(self, x):
        """ Flatten data and add bottleneck (Dense Layer).

            Arguments:
                encoder_input: A tensor.
            Returns:
                A tensor
        """
        self._shape_before_bottleneck = K.int_shape(x)[1:] # Ignore the first element (bqtch size)
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name="encoder_output")(x)
        return x


if __name__ == "__main__":
    autoencoder = AutoEncoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )

    autoencoder.summary()

