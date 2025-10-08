# 🐱🐾 Image - Text Animal Classification

## This part combines two different fundamental approaches, computer vision, and natural language processing.
## The main goal is to identify animal in both images and text.

1. Image Classification Model - EfficientNetB0 network, trained on the [**Animal10 dataset**](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data) to classify 10 categories of animals.

2. NER Model - a fine-tuned BERT trained on synthetic data to extract animals from text.


## Installation requirements:
1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv\Scripts\activate 
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## How to open the EDA file 'EDA.ipynb':
1. 
    ```bash
    jupyter notebook EDA.ipynb
    ```

## Project Structure
```bash
├── image_classification/
│   ├── examples/
│   │   ├── **/*.png
│   ├── inference_image_classification.py
│   ├── train_image_classification.py
│   ├── models
│   │   ├── **/*.h5
│   └── logs
├── ner_classification/
│   ├── models/
│   ├── train_bert.py
│   └── infer_bert.py
├── pipeline.py
├── README.md
├── EDA.ipynb
└── .gitignore
```

## Models Overview
1. EfficientNetB0 - model with custom-made top layers for classification. Classifies 10 categories:
   * dog
   * horse
   * elephant
   * butterfly
   * chicken
   * cat
   * cow
   * sheep
   * squirrel
   * spider

    The EfficientNet architecture is represented by a set of ready-to-use models. The choice depends on the required accuracy, available training resources, and input image resolution. Models are labelled from B0 (simplest) to B7 (most powerful).
    
    Graph showing the dependence of the accuracy of different models on the number of parameters.
    
    We'll use **EfficientNet-B0** architecture with the weights from **ImageNet** dataset. EfficientNet-B0 is trained on the ImageNet dataset and can classify images into 1000 object categories. The point is to re-use the weights from the pre-trained models downloading and using directly or integrating into a new model that we'll do.
    
    The include_top parameter is set to False, so the network does not include the top layer/ output layer. We have a possibility to add our own output layer depending on our needs. (10 values)
    <img src="https://habrastorage.org/r/w1560/getpro/habr/upload_files/f37/d94/853/f37d94853cbd60999b420ee88ffcb479.png" width="600">
    <img src="https://habrastorage.org/r/w1560/getpro/habr/upload_files/998/d99/1c7/998d991c728ea168111e48fdfeff8bb4.png" width="600">
    
    ### Training the model:
    `python image_classification/train_image_classification.py --epochs 12 --batch_size 32 --target_size 256
    `
    
    ### Inferencing the model:
    `python image_classification/inference_image_classification.py --image_path image_classification/examples/cat.png --model_path "image_classification/models/final_model.h5"
    `
    
    Output example: `Predicted label: cat, Confidence: 0.9743`


2. BERT for Named Entity Recognition (NER):

The model identifies animals in text using tokens (O, "B-ANIMAL").
Synthetic data is generated using different templates.
### Training the model:
`python ner_classification/train_bert.py --num_samples 2000 --model_name bert-base-cased --output_dir ner_classification/models/ner_model_bert --num_of_epochs 5 --batch_size 16 --learning_rate 5e-5 --test_size 0.2
`    
### Inferencing the model:
`python ner_classification/infer_bert.py --test_sentence "The cat is on the table" --model_path "ner_classification/models/ner_model_bert"
`    
Output example: `Extracted animal: cat`

3. General Pipeline that compares predictions by image and text models

**Input image:**
![image.png](image_classification/examples/cat.png)

**Input text:**
"Wow! That cat is so cutyy!!"

`python pipeline.py --image_path "image_classification/examples/cat.png" --text_input "Wow! That cat is so cutyy!!"`

Output examples: `The predictions from both models are the same.`