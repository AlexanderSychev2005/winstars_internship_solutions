from transformers import BertForTokenClassification

model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(labels_list)
)
