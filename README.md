# flower-classifier-tf-pipeline
End-to-end flower species classification pipeline using TensorFlow and transfer learning on TF Flower Photos &amp; Oxford 102 datasets.
Welcome to flower-classifier-tf-pipeline! This project provides a complete, ready-to-run workflow for building a flower species classifier:

Data Preparation

Downloads and merges TensorFlow’s Flower Photos (5 classes) and Oxford Flowers 102 (102 classes)

Splits into dataset/train, dataset/val, and dataset/test (80/10/10)

Model Training

Implements transfer learning with ResNet50, DenseNet121, and MobileNetV3

Applies on-the-fly data augmentation, caching, and mixed precision for GPU acceleration

Configured with EarlyStopping to automatically determine optimal epochs

Evaluation & Reporting

Generates accuracy/loss curves, confusion matrices, and a Markdown summary

Auto-creates a PowerPoint presentation summarizing results

Clone the repo, open the Jupyter/Colab notebook, set your runtime to GPU, and run all cells—everything from raw images to trained models and slides will be generated automatically.
