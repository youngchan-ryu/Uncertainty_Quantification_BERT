import pandas as pd

def preprocess_dataset():
    df = pd.read_csv('IMDB Dataset.csv')

    # Remove the break tags (<br />)
    df['review_cleaned'] = df['review'].apply(lambda x: x.replace('<br />', ''))

    # Remove unnecessary whitespace
    df['review_cleaned'] = df['review_cleaned'].replace('\s+', ' ', regex=True)

    df['sentiment_encoded'] = df['sentiment'].apply(lambda x: 0 if x == 'negative' else 1)

    return df
