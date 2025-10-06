from pathlib import Path
import spacy
import argparse


def load_model(model_path):
    """
    Loads a spacy model from disk.
    """
    nlp = spacy.load(model_path)
    return nlp


def extract_animals(text, model_path):
    """
    Extracts animal entities from the text using the spacy model.
    """
    nlp = load_model(model_path)
    doc = nlp(text)
    animals = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ == "ANIMAL":
            key = ent.text.strip().lower()
            if key not in seen:
                seen.add(key)
                animals.append(ent.text)
    return animals


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Load a trained NER model and extract animal entities from text"
    )
    parse.add_argument(
        "--model_path",
        type=str,
        default="models/model-last",
        help="Path to the trained spacy NER model",
    )
    parse.add_argument(
        "--text",
        type=str,
        default="I love that spider!",
        help="Input text to extract animal entities from",
    )
    args = parse.parse_args()

    result = extract_animals(args.text, args.model_path)
    print("Extracted animals:", result)
