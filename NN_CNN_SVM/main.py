import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import tensorflow
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image


#---------------------preprocessing data---------------------#

def load_preprocess_images(dataset_path, class_labels):
    rgb_images = []
    grayscale_images = []
    labels = []
    num_classes = len(class_labels)
    for label in class_labels:                               
        class_dir = os.path.join(dataset_path, label)    
        for img_name in os.listdir(class_dir):          
            img_path = os.path.join(class_dir, img_name) 
            
            img = image.load_img(img_path, color_mode="rgb") 
            img = image.img_to_array(img)                     
            img = normalize(img)                             
            img = tensorflow.image.resize(img, (64, 64))     

            rgb_images.append(img)      

            grayscale_img = tensorflow.image.rgb_to_grayscale(img) 
            grayscale_images.append(grayscale_img)  

            labels.append(class_labels.index(label)) 

    labels = to_categorical(labels, num_classes) 
    return np.array(rgb_images), np.array(grayscale_images), np.array(labels) 

#------------------ SVM ------------------#
def train_svm_model(train_images, train_labels):
    train_images_flattened = train_images.reshape(train_images.shape[0], -1)
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(train_images_flattened, train_labels)
    return svm_classifier

def evaluate_svm_model(svm_model, test_images, test_labels):
    test_images_flattened = test_images.reshape(test_images.shape[0], -1)
    predicted_labels = svm_model.predict(test_images_flattened)
    cm = confusion_matrix(test_labels, predicted_labels)
    f1_avg = f1_score(test_labels, predicted_labels, average='weighted')
    return cm, f1_avg

#------------ build NN models ------------#

def build_NN_model_1(input_shape, num_classes):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_NN_model_2(input_shape, num_classes):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(256, activation='relu'),
        Dense(128, activation='tanh'),
        Dense(64, activation='sigmoid'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


#------------ build CNN model ------------#

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model



#------------- train models -------------#

def train_model(model, train_images, train_labels, validation_images, validation_labels, checkpoint, batch_size=32, epochs=10):
    history = model.fit(train_images, train_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(validation_images, validation_labels),
                        callbacks=[checkpoint])
    return history


#------ confusion matrix and f1-score ------#
def confusion_score(model, test_images, test_labels):
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    cm = confusion_matrix(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    return cm, f1

#------------ plotting ------------#

def plot_curves(history, title):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} Loss Curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} Accuracy Curves')
    plt.legend()

    plt.show()


def main():
    # Path to the dataset
    train_dataset_path = 'D:/datasetsdump/Training'
    validation_dataset_path = 'D:/datasetsdump/Validation'
    class_labels = ['male', 'female']
    num_classes = len(class_labels)

    # Load and preprocess training images
    rgb_images, grayscale_images, labels = load_preprocess_images(train_dataset_path, class_labels)
   
    # train-test split rgb training images
    rgb_train_images, rgb_test_images, rgb_train_labels, rgb_test_labels = train_test_split(rgb_images, labels, test_size=0.2, random_state=42)
   
    # train-test split grayscale training images
    grayscale_train_images, grayscale_test_images, grayscale_train_labels, grayscale_test_labels = train_test_split(grayscale_images, labels, test_size=0.2, random_state=42)

    # Load and preprocess validation dataset
    rgb_validation_images, grayscale_validation_images, validation_labels = load_preprocess_images(validation_dataset_path, class_labels)

    #-----------------------  SVM  --------------------------#

    print("\n\n\n----------- SVM -------------")
    print("Training SVM on Grayscale Images...\n")
    svm_model = train_svm_model(grayscale_train_images, grayscale_train_labels.argmax(axis=1))

    svm_cm, svm_f1 = evaluate_svm_model(svm_model, grayscale_test_images, grayscale_test_labels.argmax(axis=1))
    print("\nSVM Confusion Matrix:")
    print(svm_cm)
    print("\nSVM f1-score: ", svm_f1)
    
    #-----------------------  NN  --------------------------#
    nn_checkpoint = ModelCheckpoint(filepath='nn_best_model.keras', monitor='val_accuracy',mode='max', save_best_only=True, verbose=1)

    print("\n\n\n----------- NN -------------")
    print("\n\nTraining NN model 1 on Grayscale Images...")
    input_shape = (64, 64, 1)
    model = build_NN_model_1(input_shape, num_classes)
    nn_history1 = train_model(model, grayscale_train_images, grayscale_train_labels, grayscale_validation_images, validation_labels, nn_checkpoint)
    plot_curves(nn_history1,"Model 1 NN")
    
    print("\n\nTraining NN model 2 on Grayscale Images...")
    input_shape = (64, 64, 1)
    model = build_NN_model_2(input_shape, num_classes)
    nn_history2 = train_model(model, grayscale_train_images, grayscale_train_labels, grayscale_validation_images, validation_labels, nn_checkpoint)
    plot_curves(nn_history2,"Model 2 NN")

    
    nn_best_model = load_model('nn_best_model.keras')
    nn_cm, nn_f1 = confusion_score(nn_best_model,grayscale_test_images,grayscale_test_labels)
    
    print("\n NN confusion matrix : ")
    print(nn_cm)
    print("\n NN f1-score : ",nn_f1)

    
    #-----------------------  CNN  --------------------------#

    print("\n\n\n----------- CNN -------------")
    cnn_checkpoint = ModelCheckpoint(filepath='cnn_best_model.keras', monitor='val_accuracy',mode='max', save_best_only=True, verbose=1)
    
    print("\n\nTraining CNN on Grayscale Images...")
    input_shape = (64, 64, 1)
    grayscale_model = build_cnn_model(input_shape, num_classes)
    grayscale_history = train_model(grayscale_model, grayscale_train_images, grayscale_train_labels, grayscale_validation_images, validation_labels, cnn_checkpoint)
    plot_curves(grayscale_history, 'Grayscale CNN')

    print("\n\nTraining CNN on RGB Images...")
    input_shape = (64, 64, 3)
    rgb_model = build_cnn_model(input_shape, num_classes)
    rgb_history = train_model(rgb_model, rgb_train_images, rgb_train_labels, rgb_validation_images, validation_labels, cnn_checkpoint)
    plot_curves(rgb_history, 'RGB CNN')

    # Determine the best model based on validation accuracy
    if max(rgb_history.history['val_accuracy']) > max(grayscale_history.history['val_accuracy']):
        cnn_best_model = rgb_model
        cnn_best_history = rgb_history
        cnn_test_images = rgb_test_images
        cnn_test_labels = rgb_test_labels
        print("\nBest model is RGB CNN\n")
    else:
        cnn_best_model = grayscale_model
        cnn_best_history = grayscale_history
        cnn_test_images = grayscale_test_images
        cnn_test_labels = grayscale_test_labels
        print("\nBest model is Grayscale CNN\n")

    cnn_cm, cnn_f1 = confusion_score(cnn_best_model, cnn_test_images, cnn_test_labels)

    print("\n CNN confusion matrix : ")
    print(cnn_cm)
    print("\n CNN f1-score : ",cnn_f1,"\n\n")

    print("- BEST MODEL ACCORDING TO F1-SCORE: ")
    if cnn_f1 >= nn_f1 and cnn_f1 >= svm_f1:
        print("CNN Model")
    elif nn_f1 >= cnn_f1 and nn_f1 >= svm_f1:
        print("NN Model")
    else:
        print("SVM Model")

if __name__ == "__main__":
    main()


