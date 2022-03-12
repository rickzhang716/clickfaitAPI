import numpy as np
import pandas as pd


for i in range(1):
    input = pd.read_csv(
        'datasets/clickbait_data.csv')

    clickbait_dataset = input.sample(frac=1,replace = False,random_state = 1)
    clickbait_dataset.to_csv(f"datasets/randomized_dataset{i+1}.csv")
