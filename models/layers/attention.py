
import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self, hidden_size,
                attention_size=1,
                name=None,
                 **kwargs):
        super().__init__( **kwargs)
        self.w_kernel = self.add_variable('w_kernel', [hidden_size, attention_size])
        self.w_bias = self.add_variable('w_bias', [attention_size])
        self.bias = self.add_variable('bias', [attention_size])


    def call(self, inputs, inp_len, maxlen=150, mask=None, training=False,  **kwargs):
        """
        inp_len: length of input audio
        maxlen: audio length after downsampling(cnn(twice downsample) and maxpool), in our experiments
        the input length is 1200s, after downsampling, the sequence length is 1200//8=1500,
        (8=2*2*2, see model parameters for details).
        If you change input length and times of dowansampling,
        please reset the maxlen parameter!!!!
        """
        # In case of Bi-RNN, concatenate the forward and the backward Rnn outputs.
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)
        v = tf.sigmoid(tf.tensordot(inputs, self.w_kernel, axes=1) + self.w_bias)
        vu = tf.tensordot(v, self.bias, axes=1)
        alphas = tf.nn.softmax(vu)  #(B,T)
        if mask is not None:
            alphas = alphas*tf.cast(tf.sequence_mask(inp_len, maxlen), dtype=tf.float32)
        output = tf.reduce_sum(inputs*tf.expand_dims(alphas, -1), 1)

        return output