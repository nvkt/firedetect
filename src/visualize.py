import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import random

class WildfireVisualizer:
    def __init__(self, data_dir=None, model=None, history=None):
        """
        Initialize the visualizer.
        
        Args:
            data_dir (str): Root directory containing the dataset
            model: Trained model instance
            history: Training history object
        """
        self.data_dir = data_dir
        self.model = model
        self.history = history
        
    def plot_dataset_distribution(self, save_path=None):
        """
        Plot the distribution of images across classes.
        """
        if not self.data_dir:
            raise ValueError("Data directory not provided")
            
        fire_count = len([f for f in os.listdir(os.path.join(self.data_dir, 'fire'))
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        no_fire_count = len([f for f in os.listdir(os.path.join(self.data_dir, 'no_fire'))
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=['Fire', 'No Fire'], y=[fire_count, no_fire_count])
        plt.title('Dataset Distribution')
        plt.ylabel('Number of Images')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_sample_images(self, num_samples=5, save_path=None):
        """
        Plot sample images from each class.
        """
        if not self.data_dir:
            raise ValueError("Data directory not provided")
            
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        
        # Plot fire images
        fire_files = random.sample([f for f in os.listdir(os.path.join(self.data_dir, 'fire'))
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))], num_samples)
        for i, file in enumerate(fire_files):
            img = Image.open(os.path.join(self.data_dir, 'fire', file))
            axes[0, i].imshow(img)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Fire', pad=20)
        
        # Plot no-fire images
        no_fire_files = random.sample([f for f in os.listdir(os.path.join(self.data_dir, 'no_fire'))
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))], num_samples)
        for i, file in enumerate(no_fire_files):
            img = Image.open(os.path.join(self.data_dir, 'no_fire', file))
            axes[1, i].imshow(img)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('No Fire', pad=20)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_training_history(self, save_path=None):
        """
        Plot training history metrics.
        """
        if not self.history:
            raise ValueError("Training history not provided")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot confusion matrix for model predictions.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_prediction_confidence(self, image_path, save_path=None):
        """
        Plot model's prediction confidence for a single image.
        """
        if not self.model:
            raise ValueError("Model not provided")
            
        # Load and preprocess image
        img = Image.open(image_path)
        img_array = np.array(img.resize((224, 224))) / 255.0
        prediction = self.model.predict(np.expand_dims(img_array, axis=0))[0][0]
        
        plt.figure(figsize=(12, 4))
        
        # Plot image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Input Image')
        
        # Plot confidence
        plt.subplot(1, 2, 2)
        bars = plt.bar(['No Fire', 'Fire'], [1-prediction, prediction])
        plt.title('Prediction Confidence')
        plt.ylim(0, 1)
        
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height*100:.1f}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def generate_evaluation_report(self, y_true, y_pred, save_path=None):
        """
        Generate and optionally save a complete evaluation report.
        """
        report = classification_report(y_true, y_pred, target_names=['No Fire', 'Fire'])
        print("\nClassification Report:")
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report) 