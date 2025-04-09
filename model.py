from fastai.collab import *
from fastai.tabular.all import *
import sqlite3
import csv

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
    
def fastai_model():
  update_csv()
# path = untar_data(URLs.ML_100k) # data from MovieLens
  # ratings = pd.read_csv('rating_train.csv') # provide column names
  ratings = pd.read_csv('rating_train.csv', names=['user', 'movie', 'rating', 'timestamp'])
  ratings = ratings.drop(0)
  # Convert 'movie' column in ratings to int64 data type
  ratings['movie'] = ratings['movie'].astype(int)
  # Convert 'rating' column to float
  ratings['rating'] = ratings['rating'].astype(float)
  # Convert 'timestamp' column to datetime
  ratings['timestamp'] = pd.to_datetime(ratings['timestamp'])

  movies = pd.read_csv('processed_movie.csv')
  # Rename the 'id' column to 'movie'
  movies = movies.rename(columns={'id': 'movie'})
  #print(movies.head())

  ratings = ratings.merge(movies)
  #print(ratings.head())

  dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64) # bs - batch size
  dls.show_batch()

  n_users = len(dls.classes['user'])
  n_movies = len(dls.classes['title'])
  print('The number of unique users are',n_users,'and number of the unique movies are', n_movies,'.')

  learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))

  learn.fit_one_cycle(2, 5e-3, wd=0.1) 

  learn.model

  learn.model.u_weight.weight

  learn.model.i_bias.weight.squeeze().argsort()

  def movie_list(value):
    movie_bias = learn.model.i_bias.weight.squeeze()
    if value == 'top':
        val = True 
    else:
        val = False
        value = 'Lowest'
    idxs = movie_bias.argsort(descending=val)[:10] 
    movies = [dls.classes['title'][i] for i in idxs]
    movie_bias = learn.model.bias(movies, is_item=True)
    mean_ratings = ratings.groupby('title')['rating'].mean() # get mean rating for each movie
    movie_ratings = [(b, i, mean_ratings.loc[i]) for i,b in zip(movies,movie_bias)]
    top_rated_movies = {}
    print('---10 {} rated movies are---'.format(value))
    for i in range(0,len(movies)):
      top_rated_movies[movie_ratings[i][1]] = round(movie_ratings[i][2],2)
      print('Movie:',movie_ratings[i][1],', Mean Rating:',round(movie_ratings[i][2],2))
    # print("Stored top rated movie: ", top_rated_movies)
    return top_rated_movies

  top_rated_movies = movie_list('top')
  return top_rated_movies
  # movie_list('low')
# fastai_model()

'''
def similar_movie(title):
    movie_factors = learn.model.i_weight.weight
    idx = dls.classes['title'].o2i[title]
    distances = nn.CosineSimilarity(dim=1)(movie_factors, movie_factors[idx][None])
    idx = distances.argsort(descending=True)[0:6]
    movies = dls.classes['title'][idx]
    num = 1
    for i in movies:
        print(num,'- ', i)
        num += 1        

movie = 'Star Wars (1977)'
print('The top 5 similar movies to {} are:'.format(movie))
similar_movie(movie)

movie = "Forrest Gump (1994)"
print('The top 5 similar movies to {} are:'.format(movie))
similar_movie(movie)
'''