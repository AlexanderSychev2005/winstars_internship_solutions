import random

import spacy
from spacy.tokens import DocBin
import os
from pathlib import Path
import argparse

labels = [
    "dog",
    "horse",
    "elephant",
    "butterfly",
    "chicken",
    "cat",
    "cow",
    "sheep",
    "squirrel",
    "spider",
]

templates = [
    "{animal} is in the picture.",
    "{article} {animal} is here.",
    "I see {article} {animal}.",
    "This is {article} {animal}.",
    "Here is {article} {animal}.",
    "There is {article} {animal}.",
    "A photo of {article} {animal}.",
    "A picture of {article} {animal}.",
    "An image of {article} {animal}.",
    "{animal} is so cute.",
    "I love {article} {animal}.",
    "I like {article} {animal}.",
    "Do you see {article} {animal}?",
    "Look at the {animal}!",
    "There is a {animal} here.",
    "Can you see the {animal}?",
    "What a cute {animal}!",
    "Such a beautiful {animal}.",
    "I like the {animal}.",
    "I love that {animal}.",
    "This photo shows {animal}.",
    "Maybe it's a {animal}.",
    "I spotted a {animal} yesterday.",
    "Do you see the {animal} over there?",
    "The {animal} is very friendly.",
    "The {animal} looks happy.",
    "The {animal} is very playful.",
    "The {animal} is very curious.",
    "The {animal} is very fast.",
    "The {animal} is very strong.",
    "A {animal} is running in the park.",
    "The {animal} is sleeping peacefully.",
    "Two {animal_plural} are playing.",
    "Many {animal_plural} can be seen in this image.",
    "The {animal_plural} are grazing in the field.",
    "A group of {animal_plural} is gathered here.",
    "Several {animal_plural} are swimming in the pond.",
    "Look at those {animal_plural} flying in the sky.",
    "I saw some {animal_plural} in the garden.",
    "The {animal_plural} are making noise.",
    "A couple of {animal_plural} are resting under the tree.",
]


def pluralize(animal):
    """
    Returns the plural form of an animal name.
    """
    if animal == "sheep":
        return "sheep"
    elif animal == "fish":
        return "fish"
    elif animal.endswith("y"):
        return animal[:-1] + "ies"
    elif (
        animal.endswith("s")
        or animal.endswith("x")
        or animal.endswith("z")
        or animal.endswith("ch")
        or animal.endswith("sh")
    ):
        return animal + "es"
    else:
        return animal + "s"


def make_sentence_and_span(animal):
    """
    Generates a sentence with the animal and returns the sentence and the span for the animal in the sentence.
    The span is a tuple (sentence, (start, end)).
    """
    template = random.choice(templates)
    article = "an" if animal[0].lower() in "aeiou" else "a"

    if "{animal_plural" in template:
        animal_plural = pluralize(animal)
        sentence = template.format(animal_plural=animal_plural)
        search_animal = animal_plural
    else:
        animal = animal
        sentence = template.format(animal=animal, article=article)
        search_animal = animal

    search_sentence = sentence.lower()
    search_word = search_animal.lower()

    start = search_sentence.find(search_word)

    end = start + len(search_word)
    return sentence, (start, end)


def make_dataset(n_samples):
    """
    Generates a dataset of sentences with animal entities. The result is (sentence, (start, end), label).
    """
    examples = []

    while len(examples) < n_samples:
        animal = random.choice(labels)
        sent, span = make_sentence_and_span(animal)
        examples.append((sent, span, "ANIMAL"))
    return examples


def to_docbin(examples, output_path):
    """
    Converts the dataset to a spacy DocBin format and saves it to disk.
    :return:
    """
    nlp = spacy.blank("en")
    db = DocBin()

    for text, (start, end), label in examples:
        doc = nlp(text)
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is not None:
            doc.ents = list(doc.ents) + [span]
            db.add(doc)
    db.to_disk(output_path)


def split_and_save(train_examples, val_examples, output_dir):
    print("Generating train examples...")
    train = make_dataset(train_examples)
    print("Generating valid examples...")
    valid = make_dataset(val_examples)

    to_docbin(train, os.path.join(output_dir, "train.spacy"))
    to_docbin(valid, os.path.join(output_dir, "valid.spacy"))
    print("Data saved to", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic NER data")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory to save the data"
    )
    parser.add_argument(
        "--train_size", type=int, default=10000, help="Number of training samples"
    )
    parser.add_argument(
        "--val_size", type=int, default=4000, help="Number of validation samples"
    )
    args = parser.parse_args()

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    split_and_save(args.train_size, args.val_size, args.data_dir)
