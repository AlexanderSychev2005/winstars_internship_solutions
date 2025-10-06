import random
import torch
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

labels = ["O", "B-ANIMAL", "I-ANIMAL"]

labels_map = {label: i for i, label in enumerate(labels)}
print(labels_map)

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
    Returns the plural form of an animal name.
    """
    if animal == "sheep":
        return "sheep"
    elif animal.endswith("y"):
        return animal[:-1] + "ies"
    else:
        return animal + "s"


def make_sentence_and_labels(animal):
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

    labels = [("ANIMAL", start, end)]
    return sentence, labels


def make_dataset(n_samples):

    texts, entities = [], []

    for i in range(n_samples):
        animal = random.choice(animals)
        sentence, spans = make_sentence_and_labels(animal)
        texts.append(sentence)
        entities.append(spans)
    return texts, entities


def encode_data_bio(tokenizer, texts, entities):
    encodings = tokenizer(
        texts, truncation=True, padding=True, return_offsets_mapping=True
    )
    all_labels = []

    for i, offsets in enumerate(encodings["offset_mapping"]):
        labels = ["O"] * len(offsets)
        for ent_label, start_char, end_char in entities[i]:
            token_indices = [
                idx
                for idx, (s, e) in enumerate(offsets)
                if s >= start_char and e <= end_char
            ]
            if token_indices:
                labels[token_indices[0]] = "B-ANIMAL"
                for idx in token_indices[1:]:
                    labels[idx] = "I-ANIMAL"
        labels = [labels_map[label] for label in labels]
        all_labels.append(labels)

    encodings["labels"] = all_labels
    encodings.pop("offset_mapping")
    return encodings


texts, entities = make_dataset(2000)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
dataset_dict = encode_data_bio(tokenizer, texts, entities)
dataset = Dataset.from_dict(dataset_dict)
dataset_split = dataset.train_test_split(test_size=0.2)
train_dataset = dataset_split["train"]
val_dataset = dataset_split["test"]

print(train_dataset[0])
print(val_dataset[0])


model = BertForTokenClassification.from_pretrained(
    "bert-base-cased", num_labels=len(labels)
)

training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    save_steps=200,
    eval_steps=200,
    logging_steps=50,
    learning_rate=5e-5,
    save_total_limit=2,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./models/ner_model_bert")
