import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
from ast import literal_eval
import sqlite3
import csv
from datetime import datetime

# The IMDB movies data is available on Kaggle.com
# https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

# Function to get the highest existing userId from the CSV file
def get_highest_user_id(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        highest_user_id = max(int(row[0]) for row in reader)
    return highest_user_id

def update_csv():
    # Connect to the SQLite database
    conn = sqlite3.connect(r'C:\The-Movie-Recommendation\Project\var\main-instance\db.sqlite')
    cursor = conn.cursor()
    cursor.execute("SELECT id, ratings FROM user")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # Modify the rows variable to split movie_rating into rows with the same userId
    modified_rows = []
    highest_user_id = 671
    # movie1_id = 862     # Toy Story
    # movie2_id = 19995   # Avatar
    # movie3_id = 597     # Titanic
    # movie4_id = 11      # Star Wars
    # movie5_id = 245891  # John Wick

    with open('processed_rating.csv', 'r', newline='') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Read the header
        # Filter rows and store only rows with ID <= 671 in the output file
        with open('rating_train.csv', 'w', newline='') as output:
            csv_writer = csv.writer(output)
            csv_writer.writerow(header)  # Write the header to the output file
            for row in csv_reader:
                if row and len(row) >= 1 and int(row[0]) <= highest_user_id:
                    csv_writer.writerow(row)

    for row in rows:
        user_Id = highest_user_id + 1
        # movie_rating1 = row[1]
        # movie_rating2 = row[2]
        # movie_rating3 = row[3]
        # movie_rating4 = row[4]
        # movie_rating5 = row[5]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rating_dict = row[1]
        # Convert the string representation of dictionary to an actual dictionary
        rating_dict = eval(rating_dict)  
        # Split the rating_dict into rows
        for movie_id, rating in rating_dict.items():
            modified_rows.append([user_Id, movie_id, rating, timestamp])
        # modified_rows.append([user_Id, movie1_id, movie_rating1, timestamp])
        highest_user_id += 1

    # Open a CSV file in write mode
    with open('rating_train.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the data rows
        writer.writerows(modified_rows)


def generate_prediction(userId = 1, df_movies = pd.read_csv('processed_movie.csv', low_memory=False) ):
    update_csv()
    user_id = userId+671  # some test user from the ratings file
    # 2.1 Movies Data
    # df_movies = pd.read_csv(movie_file, low_memory=False) 
    # 2.2 Ratings Data
    df_ratings_temp = pd.read_csv('rating_train.csv', low_memory=False) 

    # The Reader class is used to parse a file containing ratings.
    # The file is assumed to specify only one rating per line, such as in the df_ratings_temp file above.
    reader = Reader()
    ratings_by_users = Dataset.load_from_df(df_ratings_temp[['userId', 'movieId', 'rating']], reader)

    # Split the Data into train and test
    train_df, test_df = train_test_split(ratings_by_users, test_size=.2)

    # train an SVD model
    svd_model = SVD()
    svd_model_trained = svd_model.fit(train_df)
    '''
    ################################## Evalution #######################################
    # Perform 10-fold cross-validation
    cross_val_results = cross_validate(svd_model_trained, ratings_by_users, measures=['RMSE', 'MAE', 'MSE'], cv=10, verbose=False)
    test_mae = cross_val_results['test_mae']
    test_rmse = cross_val_results['test_rmse']
    test_mse = cross_val_results['test_mse']

    # Create a dataframe for each evaluation metric
    df_mae = pd.DataFrame({'Fold': np.arange(1, len(test_mae) + 1), 'Mean Absolute Error': test_mae})
    df_rmse = pd.DataFrame({'Fold': np.arange(1, len(test_rmse) + 1), 'Root Mean Squared Error': test_rmse})
    df_mse = pd.DataFrame({'Fold': np.arange(1, len(test_mse) + 1), 'Mean Squared Error': test_mse})

    # Sort the dataframes by the corresponding metric in descending order
    df_mae = df_mae.sort_values(by='Mean Absolute Error', ascending=False)
    df_rmse = df_rmse.sort_values(by='Root Mean Squared Error', ascending=False)
    df_mse = df_mse.sort_values(by='Mean Squared Error', ascending=False)

    # Plot the evaluation metrics per fold
    fig, axes = plt.subplots(3, 1, figsize=(5, 9))

    sns.barplot(data=df_mae, x='Fold', y='Mean Absolute Error', color='b', ax=axes[0])
    axes[0].set_title('Mean Absolute Error')

    sns.barplot(data=df_rmse, x='Fold', y='Root Mean Squared Error', color='g', ax=axes[1])
    axes[1].set_title('Root Mean Squared Error')

    sns.barplot(data=df_mse, x='Fold', y='Mean Squared Error', color='r', ax=axes[2])
    axes[2].set_title('Mean Squared Error')

    plt.tight_layout()
    plt.show()
    '''
    ############################## Generate prediction #################################
    # predict ratings for a single user_id and for all movies
    pred_series = []
    df_ratings_filtered = df_ratings_temp[df_ratings_temp['userId'] == user_id]

    print(f'Number of ratings for that user: {df_ratings_filtered.shape[0]}')
    for movie_id, name in zip(df_movies['id'], df_movies['title']):
        # check if theuser has already rated a specific movie from the list
        rating_real = df_ratings_temp.query(f'movieId == {movie_id} & userId == {user_id}')['rating'].values[0] if movie_id in df_ratings_filtered['movieId'].values else 0
        # generate the prediction
        rating_pred = svd_model_trained.predict(user_id, movie_id, rating_real, verbose=False)
        # add the prediction to the list of predictions
        pred_series.append([movie_id, name, rating_pred.est, rating_real])
    # print the results
    df_recommendations = pd.DataFrame(pred_series, columns=['movieId', 'title', 'predicted_rating', 'actual_rating'])
    df_recommendations = df_recommendations.sort_values(by='predicted_rating', ascending=False)

    # Print the recommendations
    # print(df_recommendations,'\n')
    return df_recommendations

# Call the function
# generate_prediction()