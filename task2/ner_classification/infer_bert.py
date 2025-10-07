from transformers import BertTokenizerFast, BertForTokenClassification
import torch
import argparse


def infer_bert(text, model_path):
    """
    Infers entities from the input text using a fine-tuned BERT model.

    :param text: Input text (sentence) to extract entities from
    :param model_path: Path to the fine-tuned BERT model
    :return: List of extracted entities in the format [{"entity": entity_type, "word": entity_word}, ...]
    """
    # Loading the model and the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForTokenClassification.from_pretrained(model_path)

    id2label = model.config.id2label

    # Tokenizing and encoding the input text, transforming to a tensor
    encoding = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = (
        torch.argmax(logits, dim=-1).squeeze().tolist()
    )  # get the label with the highest probability for each token, remove extra dimensions

    tokens = tokenizer.convert_ids_to_tokens(
        encoding["input_ids"].squeeze()
    )  # convert token ids to tokens

    entities = []
    for token, pred in zip(tokens, predictions):
        if (
            token in tokenizer.all_special_tokens
        ):  # Skip special tokens, for example, [CLS], [SEP]
            continue

        label = id2label[pred]
        if label.startswith("B-"):
            current_entity = {"entity": label[2:], "word": token}
            entities.append(current_entity)

    return entities


def predict_animal(text, model_path):
    """
    Predicts the animal entity from the input text using a fine-tuned BERT model.

    :param text: Input text (sentence), which contains an animal entity
    :param model_path: Path to the fine-tuned BERT model
    :return: Extracted animal entity (word) or None if no entity is found
    """
    entities = infer_bert(text, model_path)
    animal = entities[0]["word"] if entities else None
    return animal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infer animal entity from text using a fine-tuned BERT model"
    )
    parser.add_argument(
        "--test_sentence",
        type=str,
        default="Look! I love that cat!",
        help="Input text containing an animal entity",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/ner_model_bert",
        help="Path to the fine-tuned BERT model",
    )
    args = parser.parse_args()

    animal = predict_animal(args.test_sentence, args.model_path)
    print(f"Extracted animal: {animal}")
