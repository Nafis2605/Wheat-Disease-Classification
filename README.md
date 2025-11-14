## Project Overview 

This project employs deep learning models—CNN, MobileNetV2, ResNet-50, Vision Transformer (ViT), and Convolutional Vision Transformer (CvT)—to classify various wheat diseases. Consistent training parameters are used across all models to ensure reliability and reproducibility of the results. 

## File Structure 

    CODE
    ├── CNN 
    │ ├── cnn_with_class.py # CNN model with class balancing 
    │ ├── cnn_without_class.py # CNN model without class balancing 
    ├── CvT 
    │ ├── cvt_with_class.py # CvT model with class balancing 
    │ ├── cvt_without_class.py # CvT model without class balancing 
    ├── Figure Generator 
    │ ├── figure_generator.py # Script to generate visualizations
    ├── MobileNetV2 
    │ ├── trainMobileNet_noclass.py # MobileNetV2 training script without class balancing 
    │ ├── trainMobileNet.py # MobileNetV2 training script with class balancing 
    ├── Preprocessing 
    │ ├── data_partition.py # Script to partition the dataset into train, validation, and test sets 
    │ ├── data_preprocessing.py # Main preprocessing script for data preparation 
    │ ├── jpg_conversion.py # Converts images to JPEG format 
    ├── ResNet 
    │ ├── gradcam.py # Grad-CAM implementation for visualizing ResNet decisions 
    │ ├── model.py # Figure generation from model
    │ ├── resnet_classweight.py # ResNet training script with class balancing 
    │ ├── resnet_no_class_weight.py # ResNet training script without class balancing 
    ├── ViT 
    │ ├── vit_new_with_class.py # ViT model with class balancing 
    │ ├── vit_new_without_class.py # ViT model without class balancing 
    ├── README.md # Documentation of the project

 
## Getting Started

### Environment Setup: 

- Ensure Python 3.8+ and pip are installed.  
- Dependencies: 
    - PyTorch  
    - NumPy 
    - Matplotlib 
    - Jupyter (for interactive notebooks) 
    - Scikit-learn 
    - Pillow (PIL) 
    - Torchvision (for PyTorch-based models) 
    - timm (for ViT models) 
    - Torchsummary 
    - Seaborn 
    - Pandas  
    - Pickle  

 

### Training Parameters: 

All models are uniformly configured with the following parameters: 
- Optimizer: Adam 
- Learning Rate: 0.0001 
- Loss Function: Cross-Entropy Loss (with class weights if applicable) 
- Batch Size: 32 
- Epochs: Up to 150, with early stopping based on validation loss. 
  
### Testing the Models: 

To evaluate a model, ensure the trained model weights are loaded and run the model against the test dataset. Then load the reserved test dataset to compute performance metrics such as accuracy, precision, recall, F1-score, and specificity for each class. ROC curves are generated to evaluate the models’ discriminative ability. 

### Visualizing Model Insights: 

Apply Grad-CAM to visualize which parts of the images are most influential for the model's predictions. This can be particularly helpful for understanding model behavior and diagnosing issues with model learning. 

 