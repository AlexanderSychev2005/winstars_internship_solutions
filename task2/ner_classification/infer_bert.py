from transformers import BertTokenizerFast, BertForTokenClassification
import torch


labels = ["O", "B-ANIMAL", "I-ANIMAL"]
labels_map = {i: label for i, label in enumerate(labels)}


def infer_bert(text, model_path):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForTokenClassification.from_pretrained(model_path)
    encoding = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze())

    entities = []
    current_entity = None
    for token, pred in zip(tokens, predictions):
        label = labels_map[pred]
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"entity": label[2:], "word": token}
        elif label.startswith("I-") and current_entity:
            current_entity["word"] += token.replace("##", "")
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities


def predict_animal(text, model_path):
    entities = infer_bert(text, model_path)
    animal = entities[0]["word"] if entities else None
    return animal


if __name__ == "__main__":
    model_path = "./models/ner_model_bert"
    test_sentence = "I love that spider!"
    animal = predict_animal(test_sentence, model_path)
    print(f"Extracted animal: {animal}")
