import tensorflow as tf
import numpy as np

# How does the encoder work?
# 1. The encoder takes its input as a sentence in the training set.
# 2. It then converts this matrix from the embedding space to the 
# kernel size (hidden) space.
# 3. The convolutional layer runs on this matrix and we get a matrix
#  with twice the input dimensionality. We then take the GLU activation 
# function of this vector.
# 4. Step 3 is run as many times as there are convolutional layers in 
# the network.
# 5. The resultant matrix is then converted into the embedded space. 
# This is the output of the encoder
# 6. The output of the residual layer (the input to the encoder is the 
# output of the residual layer) is then added to the output of the 
# encoder to get the encoder attention

# The code below will simply be divided in terms of the steps given above

class ConvEncoder():
    def __init__(self, max_length, hidden_size, embedding_size, num_layers, dropout, is_training = True):

        # Initlialize the class variables and the placeholders for all inputs.
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.is_training = is_training
        self.max_length = max_length
        self.kernel_size = [3, self.hidden_size]
        with tf.variable_scope("ConvS2S", reuse = tf.AUTO_REUSE):
            self.X = tf.placeholder(dtype = tf.float32, shape = [None, self.max_length, self.embedding_size], name = "Encoder_Input")
            self.dense_layer_1 = tf.Variable(tf.truncated_normal([self.embedding_size, self.hidden_size], mean = 0, stddev = 1/np.sqrt(self.embedding_size)), name = "Layer_1_Encoder")
            self.dense_layer_2 = tf.Variable(tf.truncated_normal([self.hidden_size, self.embedding_size], mean = 0, stddev = 1/np.sqrt(self.embedding_size)), name = "Layer_2_Encoder")

    def for_encoder(self):
        with tf.variable_scope("ConvS2S", reuse = tf.AUTO_REUSE):

            # Step 2:
            self.X = tf.nn.dropout(self.X, keep_prob = self.dropout)
            temp = tf.reshape(self.X, [tf.shape(self.X)[0]*self.X.shape[1], self.X.shape[2]])
            dl1_out_ = tf.matmul(temp, self.dense_layer_1, name = "Layer_1_MatMul_enc")
            dl1_out_ = tf.reshape(dl1_out_, [tf.shape(dl1_out_)[0]/self.max_length, self.max_length, self.hidden_size])
            layer_output = dl1_out_
            for _ in range(self.num_layers):

                # Step 3:
                residual_output = layer_output
                self.checker = dl1_out_
                dl1_out = tf.nn.dropout(dl1_out_, keep_prob = self.dropout)
                dl1_out = tf.expand_dims(dl1_out_, axis = 0)
                self.conv_layer = tf.layers.conv2d(dl1_out, 2 * self.hidden_size, self.kernel_size, padding = "same", name = "Conv_Layer_Encoder")
                glu_output = self.conv_layer[:, :, :, 0:self.hidden_size] * tf.nn.sigmoid(self.conv_layer[:, :, :, self.hidden_size:(2*self.hidden_size)])
                glu = tf.squeeze(glu_output, axis = 0)
                layer_output = (glu + residual_output) * np.sqrt(0.5)
            
            # Step 5:
            layer_output = tf.reshape(layer_output, [tf.shape(layer_output)[0]*layer_output.shape[1], layer_output.shape[2]])
            self.encoder_output_ = tf.matmul(layer_output, self.dense_layer_2, name = "Layer_2_MatMul_enc")
            self.encoder_output_ = tf.reshape(self.encoder_output_, [tf.shape(self.encoder_output_)[0]/self.max_length, self.max_length, self.encoder_output_.shape[1]])

            # Step 6:
            self.encoder_attention_ = (self.encoder_output_ + self.X) * np.sqrt(0.5)
        return self.encoder_output_, self.encoder_attention_

        


