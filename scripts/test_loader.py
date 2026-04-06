from src.data_loader import load_reviews


def main() -> None:
    df = load_reviews()
    print(df.head())
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())


if __name__ == "__main__":
    main()