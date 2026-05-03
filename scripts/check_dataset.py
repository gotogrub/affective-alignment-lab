from datasets import load_dataset


def main() -> None:
    dataset = load_dataset("AiLab-IMCS-UL/go_emotions-en")

    print(dataset)
    print()
    print("Columns:")
    print(dataset["train"].column_names)
    print()
    print("First sample:")
    print(dataset["train"][0])


if __name__ == "__main__":
    main()