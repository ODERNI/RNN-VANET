# RNN-VANET Routing Protocol

This repository contains the code, dataset, and pretrained model for the RNN-based VANET routing protocol described in the associated research paper.

---

## Contents

- `dataset.txt`  
  Contains network simulation data used for training and evaluating the model.

- `RNN-VANET.ipynb`  
  Jupyter Notebook that implements the training of the Recurrent Neural Network (RNN) for VANET routing. Includes data preprocessing, model definition, training, and evaluation.

- `rnn-vanet.pth`  
  Pretrained PyTorch model weights saved after training. Can be loaded for inference or fine-tuning.

---

## Requirements and Environment Setup

The project requires the following Python libraries and tools:

- **Standard Libraries:**  
  `sys`, `time`, `warnings`

- **Data Handling and Visualization:**  
  `numpy`, `pandas`, `matplotlib`, `seaborn`, `matplotlib_inline` (for SVG plots in Jupyter notebooks)

- **Machine Learning Frameworks:**  
  - `PyTorch` (core deep learning framework, including modules for neural networks and data loading)  
  - `TensorFlow` (optional or for comparison)  
  - `scikit-learn` (metrics, train-test split, stratified K-Folds for balanced datasets)  
  - `iterstrat` (for multilabel stratified K-Folds)

- **Miscellaneous:**  
  - `colorama` (for colored terminal outputs during debugging)  
  - `numba` (for GPU acceleration checks)

### Debug Mode  
The code includes a debug flag (`debug = True`) to enable verbose logging and detailed outputs during training and development.

---

### Installing Dependencies

To install all necessary packages, use the following command:

```bash
pip install torch torchvision matplotlib seaborn numpy pandas scikit-learn tensorflow colorama iterstrat numba
