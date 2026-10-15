# Animal Object Detection with CNN

A computer vision project that predicts **bounding-box coordinates** for animals in images using a convolutional neural network (CNN) built with Keras. This is a bounding-box regression project, **not** an animal species classifier.

## Features
- Load JPEG images and matching XML annotations
- Resize images to 300 × 300, convert to grayscale and normalize pixel values
- Scale bounding-box coordinates to the resized image
- Train a CNN to regress `(xmin, ymin, xmax, ymax)`
- Visualize predicted boxes (red) versus annotated boxes (green)

## Repository structure
```text
notebooks/Animal_Detection_CNN.ipynb
requirements.txt
.gitignore
README.md
```

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Place your dataset in `Cats_Dogs/` at the **repository root**, then run the notebook from the repository root. Each `cat.0.jpg` or `dog.0.jpg` image must have a matching XML file such as `cat.0.xml`, containing `xmin`, `ymin`, `xmax`, and `ymax` tags. The dataset is excluded from Git.

## Model
Convolution layers (32, 64, 128 filters) with max pooling, followed by dense layers (128, 64, 32) and a 4-unit linear output. Trained using Adam (`learning_rate=0.001`) and mean absolute error (MAE) for 10 epochs.

## Evaluation and limitations
The notebook uses a 70/15/15 train/validation/test split and visualizes an example prediction. It does **not** yet report a quantitative test-set metric, IoU, or mAP. The training notebook should be run with your own data to reproduce results. Dataset source and license should be documented before public redistribution.

## Future improvements
- Report test MAE and intersection-over-union (IoU)
- Validate and clip predicted box coordinates
- Add reproducible environment versions and a dataset citation
- Compare with a pretrained object detector
