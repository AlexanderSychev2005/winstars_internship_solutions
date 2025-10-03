# Task 1 MNIST Classification with TensorFlow and Scikit-learn
In this task we're gonna build 3 models for classification of handwritten digits from the famous MNIST dataset. The MNIST dataset consists of 60,000 training images and 10,000 test images. Our classes are the digits 0-9. 

Each model is a separate class that implements the methods of MnistClassifierInterface with 2 abstract methods: **train** and **predict**.
Also, each model is hidden in MnistClassifier class that combines different patterns like Abstract Factory and Strategy.

There are 3 algorithms we're gonna build:
## Algorithms used:
1. Neural Network
2. Convolutional Neural Networks (CNN)
3. Random Forest

## How to run the code:
1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv\Scripts\activate 
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the demo file `demo.py` and run it:
   ```bash
   jupyter notebook demo.ipynb
   ```
4. Follow the instruction in the notebook. There you can train and test each model.