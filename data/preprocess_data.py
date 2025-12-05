import pandas as pd

def load_and_save_data():
    df = pd.read_csv("./data/yelp_raw.csv", usecols=['text', 'stars', 'cool', 'useful', 'funny'])
    df.to_csv("./data/yelp_preprocessed.csv", index=False)
    return df   

if __name__ == "__main__":
    load_and_save_data()