import numpy as np
import pandas as pd
import os, cv2, zipfile, shutil, h5py

import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'retina'

from tqdm import tqdm
from os.path import isfile, isdir
from sklearn.utils import shuffle

# Set random seed for Keras
np.random.seed(42)
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.applications import *

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

def visualize_epochs(epoch, history, title):
    """ Visualize trainning history by epoches
    """
    fig = plt.figure(figsize=(8, 5))

    y1 = fig.add_subplot(111)
    p1 = y1.plot(epoch, history['val_loss'], label="val_loss", color='r')
    y1.scatter(epoch, history['val_loss'], marker='*', color='r')
    y1.set_ylabel('binary_crossentropy')
    y1.set_xlabel('epochs')

    y2 = y1.twinx()
    p2 = y2.plot(epoch, history['val_acc'], label="val_acc", color='b')
    y2.scatter(epoch, history['val_acc'], marker='*', color='b')
    y2.set_ylabel('accuracy')

    lns = p1+p2
    labs = [l.get_label() for l in lns]
    y1.legend(lns, labs, loc='center right')
    plt.title(title)

    plt.show()
    
def save_test_to_csv(y_pred, name, test_gen):
    """ Save the prediction result into CSV on test data
    """
    y_pred = y_pred.clip(min=0.0045, max=0.9955)

    df = pd.read_csv("sample_submission.csv")
    for i, fname in enumerate(test_gen.filenames):
        index = int(fname.split('/')[1].split('.')[0])
        df.set_value(index-1, 'label', y_pred[i])

    outfile = 'pred_'+name+'.csv'
    print('Saving test result on: '+outfile)
    df.to_csv(outfile, index=None)
    return df

class TransferLearning_CNN(object):
    """ CNN model based on Transfer learning.
    """
    def __init__(self, train_dir, val_dir, test_dir,
                 model_name, fine_tune_layer, epochs, patience,
                 batch_size=128, class_mode='binary', data_aug=True, img_sz=(224,224)):        
        # Set random seed
        np.random.seed(42)
        
        # Initialize base_model related variables
        if model_name == 'resnet50':
            self.preproc_input = imagenet_utils.preprocess_input
            self.base = resnet50.ResNet50
        elif model_name == 'xception':
            self.preproc_input = xception.preprocess_input
            self.base = xception.Xception
        elif model_name == 'inception_res_v2':
            self.preproc_input = inception_resnet_v2.preprocess_input
            self.base = inception_resnet_v2.InceptionResNetV2
        else:
            print('Error input: Invalid input base_model!')
            return
        
        # Other variables
        self.epochs = epochs
        self.patience = patience
        self.base_name = model_name
        self.img_sz = img_sz
        self.result = None
        self.y_pred = None
        self.train_gen_full = None
        
        # Directories for image generator
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        
        # Train and Validation Image generator
        if data_aug == True:
            img_generator = ImageDataGenerator(
                                preprocessing_function=self.preproc_input,
                                rotation_range = 10,
                                zoom_range = 0.1,
                                width_shift_range = 0.05,
                                height_shift_range = 0.05,
                                channel_shift_range=10,
                                horizontal_flip=True)
        else:
            img_generator = ImageDataGenerator(preprocessing_function=self.preproc_input)
            
        self.train_gen = img_generator.flow_from_directory(
                                self.train_dir, 
                                target_size=img_sz, 
                                batch_size=batch_size,
                                shuffle=True,
                                class_mode=class_mode)

        self.val_gen = img_generator.flow_from_directory(
                                self.val_dir, 
                                target_size=img_sz, 
                                batch_size=batch_size,
                                shuffle=True,
                                class_mode=class_mode)
        
        # Test Image generator
        test_generator = ImageDataGenerator(preprocessing_function=self.preproc_input)

        self.test_gen = test_generator.flow_from_directory(
                                self.test_dir, 
                                target_size=img_sz, 
                                batch_size=batch_size,
                                shuffle=False,
                                class_mode=None)
    
        # Create model
        # Input tensor
        input_tensor = Input((img_sz[0], img_sz[1], 3))

        # base model
        self.base_model = self.base(input_tensor=input_tensor, weights='imagenet', include_top=False)

        # freeze all convolutional layers
        for layer in self.base_model.layers:
            layer.trainable = False
        # fine tune layer
        for layer in self.base_model.layers[-fine_tune_layer:]:
            layer.trainable = True

        # add a global spatial average pooling layer
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)

        # and a logistic layer
        predictions = Dense(1, activation='sigmoid')(x)

        # this is the model we will train
        self.model = Model(inputs=input_tensor, outputs=predictions)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def model_summary(self, graphic_disp=False):
        if graphic_disp == True:
            SVG(model_to_dot(self.model, show_shapes=True).create(prog='dot', format='svg'))
        else:
            self.model.summary()
        
    def model_fit(self):
        check_pt = ModelCheckpoint(
            self.base_name+'_{epoch:02d}_{val_loss:.4f}.hdf5', 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=False, 
            save_weights_only=False, 
            period=1)

        early_stop = EarlyStopping(
            monitor='val_loss', 
            min_delta=0.002,
            patience=self.patience, 
            verbose=1, 
            mode='auto')
        
        self.result = self.model.fit_generator(
                            self.train_gen,
                            steps_per_epoch=len(self.train_gen),
                            epochs=self.epochs,
                            validation_data=self.val_gen,
                            validation_steps=len(self.val_gen),
                            callbacks=[check_pt, early_stop])
        
        return self.result
        
    def model_pred(self):
        self.y_pred = self.model.predict_generator(self.test_gen, verbose=1)
        return

    def find_best_model(self):
        f = [fname for fname in os.listdir('./') if self.base_name in fname and 'hdf5' in fname]
        val_loss = list(map(lambda x:int(x), [name.split('.')[-2] for name in f]))

        sorted_models = sorted(zip(val_loss, f), key=lambda x:x[0], reverse=False)
        print('The best model is: '+sorted_models[0][1])
        return sorted_models[0][1]
        
    def load_model(self, best_model):
        # restore both weights and model arch.
        self.model = load_model(best_model)
        
    def save_test_csv(self):
        if self.y_pred.all() == None:
            print('You need to use model_pred method first!')
            return
        
        save_test_to_csv(self.y_pred, self.base_name, self.test_gen)
        
    def save_training_log(self):
        if self.result == None:
            print('You need to fit your model first!')
            return

        history = self.result.history
        train_acc = history['acc']
        train_loss = history['loss']

        val_acc = history['val_acc']
        val_loss = history['val_loss']
        
        file = 'train_log_'+self.base_name+'.npz'
        print('Saving trainning history to file: '+file)
        np.savez(file, acc=train_acc, loss=train_loss, val_acc=val_acc, val_loss=val_loss)

    def load_training_log(self):
        history = {}
        file = 'train_log_'+self.base_name+'.npz'
        print('Restoring trainning history from file: '+file)
        r = np.load(file)
        history['acc'] = r['acc']
        history['loss'] = r['loss']
        history['val_acc'] = r['val_acc']
        history['val_loss'] = r['val_loss']
        epoch = list(range(0, len(r['acc'])))
        
        return (epoch, history)
    
    def visualize_trainning(self, epoch=None, history=None):
        if self.result == None and epoch == None:
            print('You need to fit your model first!')
            return
        
        epoch = list(map(lambda x:x+1, result.epoch))
        history = self.result.history
        visualize_epochs(epoch, history, 'Trainning History of '+self.base_name)
        
    def get_full_train_generator(self):
        return self.train_gen_full
    
    def get_test_generator(self):
        return self.test_gen
        
    # high level feature extractor
    def hl_feature_extractor(self, train_full_dir, feature_enh=False):

        model = Model(self.model.input, self.model.layers[-3].output)
        print('The output of model: ', model.output)

        if feature_enh == True:
            print('Data augmentation')
            gen = ImageDataGenerator(
                preprocessing_function=self.preproc_input,
                rotation_range = 10,
                zoom_range = 0.1,
                width_shift_range = 0.05,
                height_shift_range = 0.05,
                channel_shift_range=10,
                shear_range=5,
                horizontal_flip=True,
            )
        else:
            print('Non Data augmentation')
            gen = ImageDataGenerator(preprocessing_function=self.preproc_input)

        train_gen = gen.flow_from_directory(
            train_full_dir, 
            target_size=self.img_sz, 
            shuffle=False, 
            batch_size=128,
            class_mode=None,
        )

        self.train_gen_full = train_gen
        test_gen = self.test_gen

        print('Gen feature from train data ...' )
        train = model.predict_generator(train_gen, verbose=1)

        if feature_enh == False:
            print('Gen feature from test data ...' )
            test = model.predict_generator(test_gen, verbose=1)

        fn = "feature_enh_%s.h5"%self.base_name if feature_enh==True else "feature_%s.h5"%self.base_name
        print('Write feature to file: '+fn)
        with h5py.File(fn) as h:
            h.create_dataset("train", data=train)
            h.create_dataset("label", data=train_gen.classes)
            if feature_enh==False:
                h.create_dataset("test", data=test)