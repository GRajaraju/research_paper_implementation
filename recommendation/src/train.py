import pandas as pd
import torch
import torch.nn as nn
from sklearn import model_selection


class Dataset:
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings


    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, item):
        users = self.users[item]
        movies = self.movies[item]
        ratings = self.ratings[item]

        return {
            "users" : torch.tensor(users, dtype=torch.long),
            "movies" : torch.tensor(movies, dtype=torch.long),
            "ratings" : torch.tensor(ratings, dtype=torch.float)
        }
        
def train(data_path):
    df = pd.read_csv(data_path)
    df_train, df_valid = model_selection.train_test_split(
        df, test_size=0.1, random_state=42, stratify=df.rating.values
    )

    train_dataset = Dataset(
        users=df_train.user.values, movies=df_train.movie.values,
        ratings=df_train.rating.values
    )

    valid_dataset = Dataset(
    users=df_valid.user.values, movies=df_valid.movie.values,
    ratings=df_valid.rating.values
    )

