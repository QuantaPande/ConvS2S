# ConvS2S
An implemention of the ConvS2S machine translation model using Tensorflow.

1. algo_test.py is the main file which you might want to run. This is just an experimental setup and hence, has not been coded with a specific database or dataset in mind. However, only minor changes are needed to extend this to entire language corpus.
2. How the code works is explained inside of each file separately, as I thought that would be more easier to understand. Hence, there is no point in including everything in the readme.
3. The main file is algo_test.py. The training module is the training.py file. The embedder is written in the embedding.py file. The convolutional encoder is in the conv_encoder.py file. The convolutional decoder is in the conv_decoder.py file. 
4. You might want to change the number of layers (currently 1 for both encoder and decoder) and increase the number of epochs (currently 100) for a better result. You can change the learning rate to get better results.