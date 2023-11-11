import numpy as np
import os
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from Data_parser import *
from keras.utils import to_categorical


#Define Training Data
X_train, Y_train, X_test, Y_test = read_students()
Y_train = to_categorical(Y_train, num_classes=8)
Y_test = to_categorical(Y_test, num_classes=8)

#Define grading categories


'''Create a tensorflow model'''
def createModel():
        model = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(20, activation='relu'),
                tf.keras.layers.Dense(8, activation='softmax')
        ])

        return model


'''Train the Model'''
def trainModel(model,epochs):
    checkpoint= tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only = True,
            mode='max',
            verbose = 0)
    
    #early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, mode='min',verbose=1)

    history = model.fit(
        X_train, Y_train,
        epochs = epochs,
        batch_size=15,
        validation_data=(X_test, Y_test),
        callbacks=[checkpoint])
    return history

'''Plot the accuracy and loss'''
def plotaccuracyandloss(history,epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(epochs)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
'''predict on new data'''
def predict(data):
    data = np.array([data])
    
    grades = { 0:'F', 1:'DD', 2:'CD',3:'C',4:'BC',5:'B',6:'AB',7:'A'}

    model = tf.keras.models.load_model('best_model.h5')
    predictions = model.predict(data)
    
    
    max = 0
    index = 0
    for i,x in enumerate(predictions[0]):
        if x > max:
            max = x
            index = i
    
    return grades[index]

    
    # test1 = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    # predict(test1)
    
    
    # # #Create the model
    # model = createModel()
    # model = tf.keras.models.load_model('best_model.h5')
    
    
    
    
    
    #Compile the model
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # epochs = 1000
    
    # history = trainModel(model, epochs)
    # plotaccuracyandloss(history, epochs)

    # best_model = tf.keras.models.load_model('best_model.h5')
    # loss, accuracy = best_model.evaluate(X_test, Y_test)
    
    # print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    



