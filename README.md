# Cheese Classification Challenge

This repository contains the implementation developed by Lucas Mebille and Adib Mellah for the INF473V Class Cheese Classification Challenge at Ecole polytechnique. The project focuses on classifying various types of cheese using machine learning models trained on synthetic data generated with tools like Stable Diffusion, SD-XL, and DreamBooth.

## Implemented Strategies

- **Synthetic Data Generation**: Leveraged advanced generative models like Stable Diffusion, SD-XL, and DreamBooth to create diverse and high-quality training datasets tailored for cheese classification.
- **Training Pipeline**: Developed scripts for training models on synthetic datasets, including support for fine-tuning and hyperparameter optimization using Optuna.
- **OCR Integration**: Implemented OCR algorithms to extract textual features, enhancing the classification process.
- **Data Augmentation**: Designed utilities to augment datasets, improving model robustness and addressing class imbalances.
- **Zero-Shot Learning**: Explored zero-shot approaches using CLIP to automate the selection of high-quality synthetic images.

## Libraries and Tools Used

- **Core ML/DL Frameworks**: PyTorch, TorchVision, Transformers
- **Generative Models**: Diffusers, Stable Diffusion, SD-XL, DreamBooth
- **OCR**: PyTesseract, EasyOCR
- **Data Manipulation**: Pandas, NumPy
- **Configuration and Logging**: Hydra, WandB, OmegaConf
- **Visualization**: Matplotlib, Seaborn
- **Image Augmentation**: Albumentations
- **Zero-Shot Learning**: OpenCLIP
