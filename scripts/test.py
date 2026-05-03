from datasets import load_dataset

dataset = load_dataset("AiLab-IMCS-UL/go_emotions-en")
print(dataset)
print(dataset["train"][0])