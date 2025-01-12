import tensorflow as tf
from keras.src.models import Model
from keras.src.layers import Dense, GRU
from keras.src.optimizers import AdamW

class StockModel(Model):
    """
    Class StockModel inherits the base model Keras.models.Model

    @params to __init__:
    - learning_rate: learning rate of the model
    - num_layers: the number of GRU layers
    - size_layers: the number of unit to pass to keras.layers.GRU
    - output_size: the number of unit in the output
    - forget_bias: serving dropout in recurrence 

    You need to re-implement the following methods:
    @call
    @compute_loss
    @train_step
    """
    def __init__(self, learning_rate, num_layers, size_layer, output_size, forget_bias = 0.1):
        super(StockModel, self).__init__()
        self.num_layers = num_layers
        self.size_layer = size_layer
        self.learning_rate = learning_rate
        self.rnn_cells = [
            GRU(
                units=size_layer,
                return_sequences=True if i < self.num_layers - 1 else False,
                return_state=True,
                recurrent_initializer="glorot_uniform",
                recurrent_dropout=1-forget_bias
            ) for i in range(self.num_layers)
        ]

        self.dense = Dense(units=output_size)
        self.optim = AdamW(self.learning_rate)
    
    def call(self, inputs, hidden_state):
        x = inputs
        new_states = []
        for i in range(self.num_layers):
            x, state = self.rnn_cells[i](x, initial_state=hidden_state[i])
            new_states.append(state)

        logits = self.dense(x)
        return logits, new_states
    
    def compute_loss(self, predictions, targets):
        return tf.reduce_mean(tf.square(predictions - targets))
    
    def train_step(self, inputs, targets, hidden_state):
        with tf.GradientTape() as tape:
            predictions, state = self(input, hidden_state)
            loss = self.compute_loss(predictions, targets)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optim.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss, state
