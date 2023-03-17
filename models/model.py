import tensorflow as tf
from featurizers.speech_featurizers import SpeechFeaturizer
from .layers.attention import Attention


L2 = tf.keras.regularizers.l2(1e-6)


def shape_list(x, out_type=tf.int32):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def merge_two_last_dims(x):
    b, _, f, c = shape_list(x)
    return tf.reshape(x, shape=[b, -1, f * c])


class MulSpeechLR(tf.keras.Model):
    def __init__(self, name, filters, kernel_size, d_model, rnn_cell, seq_mask, vocab_size, dropout=0.5):
        super(MulSpeechLR, self).__init__()
        self.filters1 = filters[0]
        self.filters2 = filters[1]
        self.filters3 = filters[2]
        self.kernel_size1 = kernel_size[0]
        self.kernel_size2 = kernel_size[1]
        self.kernel_size3 = kernel_size[2]
        #during training, self.mask can be set true, but during inference, it must be false
        self.mask = seq_mask
        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters1, kernel_size=self.kernel_size1,
                    strides=(2,2), padding='same', activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))
        
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters2, kernel_size=self.kernel_size2,
                    strides=(2,2), padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=self.filters3, kernel_size=self.kernel_size3,
                    strides=(1,1), padding='same', activation='relu')
        self.ln1 = tf.keras.layers.LayerNormalization(name=f"{name}_ln_1")
        self.ln2 = tf.keras.layers.LayerNormalization(name=f"{name}_ln_2")
        self.ln3 = tf.keras.layers.LayerNormalization(name=f"{name}_ln_3")
        # self.linear1 = tf.keras.layers.Dense(d_model*2, name=f"{name}_dense_1")
        self.linear2 = tf.keras.layers.Dense(d_model, name=f"{name}_dense_2")
        self.rnn = tf.keras.layers.GRU(rnn_cell, return_sequences=True, return_state=True, name=f"{name}_gru")
        self.attention = Attention(rnn_cell)
        self.class_layer = tf.keras.layers.Dense(vocab_size)
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")
        

    def call(self, inputs):
        x, x_len = inputs
        # mask = tf.cast(tf.sequence_mask(x_len, maxlen=150), dtype=tf.float32)
        x = tf.expand_dims(x, axis=-1)
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.ln2(x)
        x = self.conv3(x)
        x = self.ln3(x)
        x = merge_two_last_dims(x)
        x, final_state = self.rnn(x)
        x = self.attention(x, x_len, self.mask)
        x = self.res_add([x, final_state])
        output = self.linear2(x)
        output = tf.nn.relu(output)
        output = self.class_layer(output)
       
        return output


    def init_build(self, input_shape):
        x = tf.keras.Input(shape=input_shape, dtype= tf.float32)
        l = tf.keras.Input(shape=[], dtype=tf.int32)
        self([x, l],  training=False)

    def add_featurizers(self,
                        speech_featurizer: SpeechFeaturizer):
        """
        Function to add featurizer to model to convert to end2end tflite
        Args:
            speech_featurizer: SpeechFeaturizer instance
        """
        self.speech_featurizer = speech_featurizer


    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.float32)])
    def predict_pb(self, signal):
        features = self.speech_featurizer.tf_extract(signal)
        input_len = tf.expand_dims(tf.shape(features)[0], axis=0)
        input = tf.expand_dims(features, axis=0)
        output = self([input, input_len], training=False)
        output = tf.nn.softmax(output)
        output1 = tf.squeeze(output)
        output = tf.argmax(output1, axis=-1)

        return output, tf.gather(output1, output)