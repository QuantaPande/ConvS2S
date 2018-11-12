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
    def __init__(self, vocab_size, max_length, hidden_size, embedding_size, num_layers, dropout, is_training = True):

        # Initlialize the class variables and the placeholders for all inputs.
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.is_training = is_training
        self.max_length = max_length
        self.kernel_size = [3, self.hidden_size]
        self.vocab_size = vocab_size
        with tf.variable_scope("ConvS2S", reuse = tf.AUTO_REUSE):
            self.dense_layer_1 = tf.Variable(tf.truncated_normal([self.embedding_size, self.hidden_size], mean = 0, stddev = 1/np.sqrt(self.embedding_size)), name = "Layer_1_Encoder")
            self.dense_layer_2 = tf.Variable(tf.truncated_normal([self.hidden_size, self.embedding_size], mean = 0, stddev = 1/np.sqrt(self.embedding_size)), name = "Layer_2_Encoder")
            self.dense_layer_3 = tf.Variable(tf.truncated_normal([self.embedding_size, self.vocab_size], mean = 0, stddev = 1/np.sqrt(self.embedding_size)), name = "Layer_3_Encoder")
            self.layer_conv_embedding = tf.Variable(tf.truncated_normal([self.hidden_size, self.embedding_size], mean = 0, stddev = 1/np.sqrt(self.embedding_size)), name = "Hid_to_Embed_att_dec")
            self.layer_embedding_conv = tf.Variable(tf.truncated_normal([self.embedding_size, self.hidden_size], mean = 0, stddev = 1/np.sqrt(self.embedding_size)), name = "Embed_to_Hid_att_dec")
            

    def for_encoder(self):
        with tf.variable_scope("ConvS2S", reuse = tf.AUTO_REUSE):
            self.X = tf.placeholder(dtype = tf.float32, shape = [None, self.max_length, self.embedding_size], name = "Encoder_Input")
            # Step 2:
            self.X = tf.nn.dropout(self.X, keep_prob = self.dropout)
            temp = tf.reshape(self.X, [tf.shape(self.X)[0]*self.X.shape[1], self.X.shape[2]])
            dl1_out_ = tf.matmul(temp, self.dense_layer_1, name = "Layer_1_MatMul_enc")
            dl1_out_ = tf.reshape(dl1_out_, [tf.shape(dl1_out_)[0]/self.max_length, self.max_length, self.hidden_size])
            layer_output = dl1_out_
            for _ in range(self.num_layers):

                # Step 3:
                residual_output = layer_output
                self.checker = layer_output
                dl1_out = tf.nn.dropout(layer_output, keep_prob = self.dropout)
                dl1_out = tf.expand_dims(dl1_out, axis = 0)
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
    
    def rev_encoder(self, decoder_inputs, decoder_attention):
        with tf.variable_scope('ConvS2S', reuse = tf.AUTO_REUSE):
            self.X_rev = tf.placeholder(dtype = tf.float32, shape = [None, self.max_length - 1, self.embedding_size], name = "Encoder_input_rev")
            self.X_rev = tf.nn.dropout(self.X_rev, keep_prob = self.dropout)
            temp = tf.reshape(self.X_rev, [tf.shape(self.X_rev)[0]*self.X_rev.shape[1], self.X_rev.shape[2]])
            dl1_out_ = tf.matmul(temp, self.dense_layer_1, name = "Layer_1_MatMul_enc_rev")
            dl1_out_ = tf.reshape(dl1_out_, [tf.shape(dl1_out_)[0]/(self.max_length - 1), self.max_length - 1, self.hidden_size])
            layer_output = dl1_out_

            for _ in range(self.num_layers):
                residual_layer = layer_output
                dl1_out = tf.nn.dropout(layer_output, keep_prob = self.dropout)
                dl1_out = tf.expand_dims(dl1_out, axis = 0)
                self.conv_layer = tf.layers.conv2d(dl1_out, 2 * self.hidden_size, self.kernel_size, padding = "same", name = "Conv_Layer_Encoder")
                glu_output = self.conv_layer[:, :, :, 0:self.hidden_size] + tf.nn.sigmoid(self.conv_layer[:, :, :, self.hidden_size:(2 * self.hidden_size)])
                glu = tf.squeeze(glu_output, axis = 0)
                layer_output = (glu + residual_layer) * np.sqrt(0.5)
                shape_out = tf.shape(layer_output)


                layer_output_ = tf.reshape(layer_output, [tf.shape(layer_output)[0]*layer_output.shape[1], layer_output.shape[2]])
                post_glu_output = tf.matmul(layer_output_, self.layer_conv_embedding, name = "Hid_to_Embedding_enc_rev_att")
                post_glu_output = tf.reshape(post_glu_output, [tf.shape(post_glu_output)[0]/(self.max_length - 1), self.max_length - 1, self.embedding_size])
                decoder_attention_logits = tf.matmul(post_glu_output, tf.transpose(decoder_attention, perm = [0, 2, 1]), name = "Decoder_Attention_MatMul")
                decoder_attention_outputs = tf.nn.softmax(decoder_attention_logits, axis = 0)


                attention_output = tf.matmul(decoder_attention_outputs, decoder_inputs, name = "Attention_Output_rev")
                attention_output = attention_output * (decoder_inputs.shape.as_list()[2] / np.sqrt(2 / decoder_inputs.shape.as_list()[2]))
                layer_output_1 = tf.reshape(attention_output, [tf.shape(attention_output)[0]*attention_output.shape[1], attention_output.shape[2]])
                layer_output_2 = tf.matmul(layer_output_1, self.layer_embedding_conv, name = "Embedding_to_Hid_enc_rev_att")
                layer_output = tf.reshape(layer_output_2, shape_out)
            
            layer_output_3 = (layer_output + dl1_out_) * np.sqrt(0.5)
            layer_output = tf.reshape(layer_output_3, [tf.shape(layer_output_3)[0]*layer_output_3.shape[1], layer_output_3.shape[2]])
            output = tf.matmul(layer_output, self.dense_layer_2, name = "Layer_2_Enc_MatMul_rev")

            self.prob_output = tf.matmul(output, self.dense_layer_3, name = "Layer_3_Enc_MatMul_rev")
            self.prob_output = tf.reshape(self.prob_output, [tf.shape(self.prob_output)[0]/(self.max_length - 1), self.max_length - 1, self.vocab_size])
            self.prob_output = tf.nn.softmax(self.prob_output, 2)
            writer = tf.summary.FileWriter("D:/College/Masters/Summer/ConvS2S/Tensorboards/" + str(np.random.randint(0, 10000)))
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                writer.add_graph(sess.graph)
        return (self.prob_output)




        


