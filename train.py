from preprocess import generating_training_sequences,SEQUENCE_LENGTH
import tensorflow.keras as keras

OUTPUT_UNITS = 38
lOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
NUM_UNITS = [256]
EPOCHS= 50
BATCH_SIZE=64

def build_model(output_units,num_units, loss, learning_rate):

    # create the model architecture
    input = keras.layers.Input(shape=(None,output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units,activation="softmax")(x)

    model = keras.Model(input,output)

    # compile model
    model.compile(loss=loss,optimizer=keras.optimizers.Adam(lr=learning_rate),metrics=["accuracy"])

    # model.summary()

    return model

def train(output_units=OUTPUT_UNITS,num_units=NUM_UNITS, loss=lOSS, learning_rate=LEARNING_RATE):

    # generate training sequences
    inputs,target = generating_training_sequences(SEQUENCE_LENGTH)

    # build the network
    model = build_model(output_units,num_units, loss, learning_rate)

    # train the model

    model.fit(inputs,target, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save("model.h5")


if __name__=="__main__":
    train()