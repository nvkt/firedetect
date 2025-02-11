import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

class WildfireDataLoader:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the data loader for wildfire detection.
        
        Args:
            data_dir (str): Root directory containing 'fire' and 'no_fire' subdirectories
            img_size (tuple): Target size for images (height, width)
            batch_size (int): Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.fire_dir = os.path.join(data_dir, 'fire')
        self.no_fire_dir = os.path.join(data_dir, 'no_fire')
        
        # Data augmentation for training
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Only rescaling for validation
        self.val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )

    def load_data_generators(self):
        """
        Create train and validation data generators.
        
        Returns:
            tuple: (train_generator, validation_generator)
        """
        train_generator = self.train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            classes=['no_fire', 'fire']
        )
        
        validation_generator = self.val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            classes=['no_fire', 'fire']
        )
        
        return train_generator, validation_generator

    def preprocess_single_image(self, image_path):
        """
        Preprocess a single image for inference.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image ready for model input
        """
        img = Image.open(image_path)
        img = img.resize(self.img_size)
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def validate_data_directory(self):
        """
        Validate that the data directory has the correct structure and contains images.
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not os.path.exists(self.fire_dir) or not os.path.exists(self.no_fire_dir):
            print(f"Error: Missing required directories in {self.data_dir}")
            return False
            
        fire_images = len([f for f in os.listdir(self.fire_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        no_fire_images = len([f for f in os.listdir(self.no_fire_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if fire_images == 0 or no_fire_images == 0:
            print(f"Error: No images found in one or both directories")
            print(f"Fire images: {fire_images}")
            print(f"No fire images: {no_fire_images}")
            return False
            
        print(f"Found {fire_images} fire images and {no_fire_images} no-fire images")
        return True 