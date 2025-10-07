import random
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import argparse
from pathlib import Path

labels = ["O", "B-ANIMAL"]  # Outside, Beginning of animal

labels_map = {label: i for i, label in enumerate(labels)}  # Mapping labels to integers
print(labels_map)

# Synthetic data generation
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

animals = [
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


def pluralize(animal):
    """
    Returns the plural form of an animal name. Handles special cases like "sheep" and words ending with "y".
    """
    if animal == "sheep":
        return "sheep"
    elif animal.endswith("y"):
        return animal[:-1] + "ies"
    else:
        return animal + "s"


def make_sentence_and_labels(animal):
    """
    Generates a sentence, which includes the given animal, and adds the start and the end indexes.
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

    labels = [(start, end)]
    return sentence, labels


def make_dataset(n_samples):
    """
    Generates a dataset with n_samples samples using the function to create sentences and labels.
    """
    texts, entities = [], []

    for i in range(n_samples):
        animal = random.choice(animals)
        sentence, spans = make_sentence_and_labels(animal)
        texts.append(sentence)
        entities.append(spans)
    return texts, entities


def encode_data_bio(tokenizer, texts, entities):
    """
    Encodes the texts and the labels using the BIO structures.
    Tokenization with offsets, offset mapping is needed to align labels with tokens, returns start and end character
    positions of each token (start, end).
    truncation - truncate sequences to the model's maximum length
    padding - add padding tokens to make all sequences the same length
    """
    encodings = tokenizer(
        texts, truncation=True, padding=True, return_offsets_mapping=True
    )
    all_labels = []

    for i, offsets in enumerate(encodings["offset_mapping"]):
        labels = []
        for start, end in offsets:
            if (
                start == 0 and end == 0
            ):  # [PAD] token, set label to -100, so it's ignored in loss computation
                labels.append(-100)
            else:
                labels.append(labels_map["O"])  # Not an entity, set label to O

        # Set labels B-ANIMAL for each entity, we don't have I-ANIMAL as entities are single tokens (single words)
        for start_char, end_char in entities[i]:
            token_indices = [
                idx
                for idx, (s, e) in enumerate(offsets)
                if s >= start_char
                and e <= end_char  # Token is inside the entity or the entity itself
            ]
            if token_indices:
                labels[token_indices[0]] = labels_map["B-ANIMAL"]

        all_labels.append(labels)

    encodings["labels"] = all_labels
    encodings.pop("offset_mapping")
    return encodings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training the NER BERT model on synthetic data with animals"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of samples to generate for training",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-cased",
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/ner_model_bert",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--num_of_epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split",
    )

    args = parser.parse_args()

    # Generating dataset
    texts, entities = make_dataset(args.num_samples)
    print(
        f"Generated {len(texts)} sentences, example: \n {texts[0]} \n labels: \n {entities[0]}."
    )

    # Tokenization and encoding the data
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    encodings = encode_data_bio(tokenizer, texts, entities)

    # Creating Dataset object and splitting into train and validation sets
    dataset = Dataset.from_dict(encodings)
    dataset_split = dataset.train_test_split(test_size=args.test_size)
    train_dataset = dataset_split["train"]
    val_dataset = dataset_split["test"]

    # Decoding example
    example = train_dataset[0]
    decoded_example = tokenizer.decode(example["input_ids"], skip_special_tokens=False)
    print(
        f"Decoded example: \n {decoded_example} \n with labels: \n {example['labels']} ."
    )

    model = BertForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(labels)
    )
    model.config.id2label = {i: label for i, label in enumerate(labels)}

    model.config.label2id = {label: i for i, label in enumerate(labels)}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_of_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        save_steps=200,
        eval_steps=200,
        logging_steps=50,
        learning_rate=args.learning_rate,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
