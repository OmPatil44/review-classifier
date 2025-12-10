import pandas as pd

PATH = r'./dataset/IMDB Dataset.csv'
df = pd.read_csv(PATH)

def split(number_of_dataset):
    total_rows = df.shape[0]
    row_length = int(total_rows / number_of_dataset)
    print(f"Total Rows : {total_rows}")
    print(f"Row Length : {row_length}")
    rows = 0
    for i in range(1,number_of_dataset + 1):
        print(f"Writing rows {rows} to {row_length}")
        # with open(file=f"./dataset/dataset_{i}.csv", mode='w') as f:
        #     f.write(str(df[rows : row_length]))
            # rows = row_length
            # row_length += 10000

        new_df = df[rows:row_length]
        new_df.to_csv(f"./dataset/dataset_{i}.csv", index=False)
        rows = row_length
        row_length += 2500

split(20)
