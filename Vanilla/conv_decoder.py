import tensorflow as tf
import numpy as np

# How does the decoder work?:
# 1. The decoder takes the decoder inputs (Which are just the targets with the 
# last element removed) and the target values (which is just the target vector 
# with the first element removed) and the encoder outputs and the encoder 
# attention
# 2. It then converts the input from the embedding space to the size of the kernel 
# (the hidden space).
# 3. The convolutional layer runs its course and we get a matrix which has 
# dimensionality twice the hidden space. We then run the GLU activation unit
# on the output. We get back a matrix in the hidden space.
# 4. This GLU output is converted back into the embedded space. The GLU output 
# is then dot multiplied with the encoder attention. (Why? That is 
# how we define the attention in the original paper) We take the softmax of 
# the resulting matrix. 
# 5. The attention output is the matrix multiplication of the encoder outputs 
# and the vector we got from the above step. We then scale this attention and 
# convert it back to hidden space.
# 6. The process steps 3-5 are repeated as per the number of layers in the 
# convolutional network. The final output of the convolutional network is 
# converted back into the embedding space.
# 7. This output is converted into vocab space, and the softmax is taken. 
# 8. This output is the final output of the network. We predict the correct 
# words based on this. 
# 9. In case of calculating the output in the forward direction, the decoder 
# inputs are a matrix of dimensions (no_of_input_sentences * (max_decoder_length + 2)). 
# We initialize the first column of this matrix with a all ones vector. We feed 
# this matrix to the decoder and we get thee next column in the decoder 
# input matrix. We keep feeding this till we get a <end> token. This is the output 
# prediction.

# The code below will simply be divided in terms of the steps given above
class ConvDecoder():
    def __init__(self, vocab_size, max_length, hidden_size, embedding_size, num_layers, dropout, is_training = True):

        # Initialize the class variables
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.p = dropout
        self.is_training = is_training
        self.kernel_size = [5, self.hidden_size]

        # Define the various placeholders and layers
        with tf.variable_scope("ConvS2S", reuse = tf.AUTO_REUSE):
            self.dense_layer_1 = tf.Variable(tf.truncated_normal([self.embedding_size, self.hidden_size], mean = 0, stddev = 1/np.sqrt(self.embedding_size)), name = "Layer_1_Dec")
            self.dense_layer_2 = tf.Variable(tf.truncated_normal([self.hidden_size, self.embedding_size], mean = 0, stddev = 1/np.sqrt(self.embedding_size)), name = "Layer_2_Dec")
            self.dense_layer_3 = tf.Variable(tf.truncated_normal([self.embedding_size, self.vocab_size], mean = 0, stddev = 1/np.sqrt(self.embedding_size)), name = "Layer_3_Dec")
            self.layer_conv_embedding = tf.Variable(tf.truncated_normal([self.hidden_size, self.embedding_size], mean = 0, stddev = 1/np.sqrt(self.embedding_size)), name = "Hid_to_Embed_att_dec")
            self.layer_embedding_conv = tf.Variable(tf.truncated_normal([self.embedding_size, self.hidden_size], mean = 0, stddev = 1/np.sqrt(self.embedding_size)), name = "Embed_to_Hid_att_dec")
            self.input_x = tf.placeholder(dtype = tf.float32, shape = [None, self.max_length - 1, self.embedding_size], name = "Decoding_Input")
        
    def decoder(self, encoder_outputs, encoder_attention):
        with tf.variable_scope("ConvS2S", reuse = tf.AUTO_REUSE):

            # Step 1:
            self.input_x = tf.nn.dropout(self.input_x, keep_prob = self.p)
            temp = tf.reshape(self.input_x, [tf.shape(self.input_x)[0]*self.input_x.shape[1], self.input_x.shape[2]])

            # Step 2:
            dl1_out_ = tf.matmul(temp, self.dense_layer_1, name = "Dec_Layer_1_MatMul")
            dl1_out_ = tf.reshape(dl1_out_,  [tf.shape(dl1_out_)[0]/(self.max_length - 1), self.max_length - 1, self.hidden_size])
            layer_output = dl1_out_
            for _ in range(self.num_layers):

                # Step 3
                residual_output = layer_output
                dl1_out = tf.nn.dropout(dl1_out_, keep_prob = self.p)
                dl1_out = tf.expand_dims(dl1_out_, axis = 0)
                self.conv_layer = tf.layers.conv2d(dl1_out, 2 * self.hidden_size, self.kernel_size, padding = "same", name = "Conv_layer_Dec")
                glu_output = self.conv_layer[:, :, :, 0:self.hidden_size] * tf.nn.sigmoid(self.conv_layer[:, :, :, self.hidden_size:(2*self.hidden_size)])
                glu = tf.squeeze(glu_output, axis = 0)
                layer_output = (glu + residual_output) * np.sqrt(0.5)
                shape_out = tf.shape(layer_output)

                # Step 4:
                layer_output = tf.reshape(layer_output, [tf.shape(layer_output)[0]*layer_output.shape[1], layer_output.shape[2]])
                post_glu_output = tf.matmul(layer_output, self.layer_conv_embedding, name = "Hid_to_Embed_att_MatMul")
                post_glu_output = tf.reshape(post_glu_output, [tf.shape(post_glu_output)[0]/(self.max_length - 1), self.max_length - 1, self.embedding_size])
                encoder_attention_logits = tf.matmul( post_glu_output, tf.transpose(encoder_attention, perm = [0, 2, 1]), name = "Encoder_Attention_MatMul")
                encoder_attention_output = tf.nn.softmax(encoder_attention_logits, axis = 2)

                # Step 5:
                attention_output = tf.matmul(encoder_attention_output, encoder_outputs, name = "Attention_Output")
                attention_output = attention_output * (encoder_outputs.shape.as_list()[2] * np.sqrt(2 / encoder_outputs.shape.as_list()[2]))
                layer_output = tf.reshape(attention_output, [tf.shape(attention_output)[0]*attention_output.shape[1], attention_output.shape[2]])
                layer_output = tf.matmul(layer_output, self.layer_embedding_conv, name = "Embed_to_hid_MatMul")
                layer_output = tf.reshape(layer_output, shape_out)
            
            # Step 6:
            layer_output = (layer_output + dl1_out_) * np.sqrt(0.5)
            layer_output = tf.reshape(layer_output, [tf.shape(layer_output)[0]*layer_output.shape[1], layer_output.shape[2]])
            output = tf.matmul(layer_output, self.dense_layer_2, name = "Layer_2_Dec_MatMul")

            # Step 7:
            self.prob_output = tf.matmul(output, self.dense_layer_3, name = "Layer_3_Dec_MatMul")
            self.prob_output = tf.reshape(self.prob_output, [tf.shape(self.prob_output)[0]/(self.max_length - 1), self.max_length - 1, self.vocab_size])
            self.prob_output = tf.nn.softmax(self.prob_output, 2)
            writer = tf.summary.FileWriter("D:/College/Masters/Summer/ConvS2S/Tensorboards/" + str(np.random.randint(0, 10000)))
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                writer.add_graph(sess.graph)
        return (self.prob_output)
