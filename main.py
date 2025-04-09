import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session, current_app, make_response
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests
from datetime import date, datetime
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectMultipleField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Session
from flask_login import LoginManager, logout_user 
from sqlalchemy import PickleType
from flask_migrate import Migrate, upgrade
import csv

# load the nlp model and tfidf vectorizer from disk
clf = pickle.load(open('nlp_model.pkl', 'rb')) # sentiment analysis
vectorizer = pickle.load(open('tranform.pkl','rb'))
    
# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

# convert list of numbers to list (eg. "[1,2,3]" to [1,2,3])
def convert_to_list_num(my_list):
    my_list = my_list.split(',')
    my_list[0] = my_list[0].replace("[","")
    my_list[-1] = my_list[-1].replace("]","")
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

app = Flask(__name__)
# DB for storing users' info.
app.config['SECRET_KEY'] = '9OLWxND4o83j4K4iuopO'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model): # SQLite database structure
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    preferences = db.Column(db.Text)
    ratings = db.Column(db.String(2000), nullable=True)
    movie1_rating = db.Column(db.Integer)
    movie2_rating = db.Column(db.Integer)
    movie3_rating = db.Column(db.Integer)
    movie4_rating = db.Column(db.Integer)
    movie5_rating = db.Column(db.Integer)

    def __init__(self, username, email, password, preferences, rating, movie1_rating, movie2_rating, movie3_rating, movie4_rating, movie5_rating):
        self.username = username
        self.email = email
        self.password = password
        self.preferences = preferences
        self.ratings = rating
        self.movie1_rating = movie1_rating
        self.movie2_rating = movie2_rating
        self.movie3_rating = movie3_rating
        self.movie4_rating = movie4_rating
        self.movie5_rating = movie5_rating


with app.app_context():
    db.create_all()

genres = ['Action', 'Adventure','Animation','Comedy','Crime','Documentary','Drama','Family','Fantasy','History','Horror','Music','Mystery','Romance'
          ,'Science Fiction','TV Movie','Thriller','War']

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    genre_preferences = SelectMultipleField('Genre Preferences', choices=[(genre, genre) for genre in genres], validators=[DataRequired()])
    movie1_rating = SelectField('Toy Story Rating', choices=[(str(i), str(i)) for i in range(1, 6)], validators=[DataRequired()])
    movie2_rating = SelectField('Avatar Rating', choices=[(str(i), str(i)) for i in range(1, 6)], validators=[DataRequired()])
    movie3_rating = SelectField('Titanic Rating', choices=[(str(i), str(i)) for i in range(1, 6)], validators=[DataRequired()])
    movie4_rating = SelectField('Star War (1977) Rating', choices=[(str(i), str(i)) for i in range(1, 6)], validators=[DataRequired()])
    movie5_rating = SelectField('John Wick Rating', choices=[(str(i), str(i)) for i in range(1, 6)], validators=[DataRequired()])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


@app.route("/")
def main_page():
    return render_template("base.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = generate_password_hash(form.password.data)
        genre_preferences = form.genre_preferences.data
        movie1_rating = form.movie1_rating.data
        movie2_rating = form.movie2_rating.data
        movie3_rating = form.movie3_rating.data
        movie4_rating = form.movie4_rating.data
        movie5_rating = form.movie5_rating.data
        # Check if the user already exists in the database
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email address already exists, please login ', 'danger')
            return redirect(url_for('register'))    
        genre_preferences=list(genre_preferences)
        genre_preferences_json=json.dumps(genre_preferences)
        ratings_dict = {}
        if movie1_rating:
            ratings_dict[862] = movie1_rating
        if movie2_rating:
            ratings_dict[19995] = movie2_rating
        if movie3_rating:
            ratings_dict[597] = movie3_rating
        if movie4_rating:
            ratings_dict[11] = movie4_rating
        if movie5_rating:
            ratings_dict[245891] = movie5_rating
        ratings_json = json.dumps(ratings_dict)
        new_user = User(username, email, password, genre_preferences_json, ratings_json, movie1_rating, movie2_rating, movie3_rating, movie4_rating, movie5_rating)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html', form=form, genres=genres)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            username = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('login.html', form=form)

from functools import wraps
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access home page', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def fetch_movie_details(movie_id):
    # Make an API call to fetch movie details using the movie_id
    api_key = '78720ad51fd22cc69b2babb6a3e4d1b4'
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'  # Replace with your API endpoint
    response = requests.get(url)
    if response.status_code == 200:
        movie_data = response.json()
        poster_path = movie_data['poster_path']
        poster = f'https://image.tmdb.org/t/p/original{poster_path}' if poster_path else 'static/default.jpg'
        movie_details = {
            'title' : movie_data['title'],
            'poster' : poster,
            'rel_date' : movie_data['release_date']
        }
        return movie_details
    else:
        return None

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

from model import fastai_model
import asyncio

@app.route("/home")
@login_required
def home():
    # Access the user's preferences from the database
    user_id = session.get('user_id')
    if user_id:
        user = User.query.get(user_id)
        if user:
            if user.preferences:
                preferences = user.preferences
                suggestions = get_suggestions() # list of movies

                # Render the loading screen template
                loading_template = render_template('loading.html')
                response = make_response(loading_template)

                # Define an asynchronous function to call the fastai_model
                async def get_recommendations():
                    recommendations = await asyncio.get_running_loop().run_in_executor(None, fastai_model)
                    movie_recommendations = {}
                    with open('processed_movie.csv', 'r', encoding='utf-8') as csvfile:
                        reader = csv.reader(csvfile)
                        for row in reader:
                            movie_id = row[5]
                            movie_title = row[0]
                            movie_rating = recommendations.get(movie_title)
                            if movie_title in recommendations:
                                movie_details = fetch_movie_details(movie_id)
                                movie_recommendations[movie_id] = {
                                    'rating': movie_rating,
                                    'title' : movie_title,
                                    'details': movie_details
                                    }
                    return movie_recommendations

                # Run the fastai_model asynchronously
                
                recommendations = asyncio.run(get_recommendations())
                print ("related movie details: ", recommendations)
                movie_cards = {
                    recommendations[movie_id]['details']['poster']: [
                        recommendations[movie_id]['details']['title'],
                        datetime.strptime(recommendations[movie_id]['details']['rel_date'], '%Y-%m-%d').year,
                        recommendations[movie_id]['rating'],
                        movie_id
                    ]
                    for movie_id in recommendations
                }

                # Update the response with the home template and recommendations
                home_template = render_template('home.html', suggestions=suggestions, username=user.username,genres=preferences, movie_cards=movie_cards)
                response.set_data(home_template)
                return response
    return redirect(url_for('login'))

# SVD_model
from SVD_model import generate_prediction

@app.route("/populate-matches",methods=["POST"])
def populate_matches():    
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    if user:
        genre_preferences = json.loads(user.preferences)
        # print('---------genres-------------\n',genre_preferences)
    # Preprocess genre preferences to convert them to numbers
    genre_map = {
        'Action': 28,
        'Adventure': 12,
        'Animation': 16,
        'Comedy': 35,
        'Crime': 80,
        'Documentary': 99,
        'Drama': 18,
        'Family': 10751,
        'Fantasy': 14,
        'History': 36,
        'Horror': 27,
        'Music': 10402,
        'Mystery': 9648,
        'Romance': 10749,
        'Science Fiction': 878,
        'TV Movie': 10770,
        'Thriller': 53,
        'War': 10752,
        'Western': 37
    }
    genre_preferences = [genre_map.get(genre.title()) for genre in genre_preferences if genre.title() in genre_map]
    # print('---------converted genres-------------\n',genre_preferences)
    # getting data from AJAX request
    res = json.loads(request.get_data("data"))
    movies_list = res['movies_list']
    # Convert the list to a DataFrame
    df_movies = pd.DataFrame(movies_list, columns=['id', 'title', 'vote_average', 'vote_count', 'genre_ids', 'release_date', 'poster_path', 'original_title'])
    # Append the new movies to the processed_movie DataFrame
    processed_movie = pd.read_csv('processed_movie.csv')
    # Create a mask to identify the missing id values
    mask = ~processed_movie['id'].isin(df_movies['id'])
    # Filter the df_movies DataFrame to include only the missing id values
    new_movies = df_movies[mask]
    new_movies = new_movies[['id', 'title']]
    print("new movies appended: ", new_movies)
    # Concatenate the new_movies DataFrame with the processed_movie DataFrame
    processed_movie = pd.concat([processed_movie, new_movies], ignore_index=True)
    # Save the updated DataFrame back to the processed_movie.csv file
    processed_movie.to_csv('processed_movie.csv', index=False)
    # Save the DataFrame to a CSV file
    df_movies.to_csv('movies_list.csv', index=False)
    df_recommendations = generate_prediction(user_id, df_movies)
    print(df_recommendations)
    # Merge the movie dataframe with the predicted ratings dataframe based on movie ID
    merged_df = pd.merge(df_movies, df_recommendations, left_on='id', right_on='movieId')
    # Define a custom sorting key function
    def custom_sort_key(row):
        if row['genre_ids'] in genre_preferences:
            return (False, -row['predicted_rating'])
        else:
            return (True, -row['predicted_rating'])
    
    # Apply the custom sorting key to the DataFrame
    merged_df['sort_key'] = merged_df.apply(custom_sort_key, axis=1)
    merged_df = merged_df.sort_values(by='sort_key')
    # merged_df = merged_df.sort_values(by='predicted_rating', ascending=False)
    rearranged_movie_list = merged_df.to_dict('records')
    print(rearranged_movie_list)
    '''
    filtered_movies = []
    other_movies = []
    for movie in rearranged_movie_list:
        genre_ids = movie.get('genre_ids', [])
        matching = False
        # print('---------genre ids -------------\n',genre_ids)
        for genre_id in genre_ids:
            if genre_id in genre_preferences:
                matching = True
                break
        if matching:
            filtered_movies.append(movie)
        else:
            other_movies.append(movie)
    shuffled_movies = filtered_movies+other_movies
    '''
    movie_cards = {"https://image.tmdb.org/t/p/original"+rearranged_movie_list[i]['poster_path'] if rearranged_movie_list[i]['poster_path'] else "/static/movie_placeholder.jpeg": [rearranged_movie_list[i]['title_x'],rearranged_movie_list[i]['original_title'],rearranged_movie_list[i]['vote_average'],datetime.strptime(rearranged_movie_list[i]['release_date'], '%Y-%m-%d').year if rearranged_movie_list[i]['release_date'] else "N/A", rearranged_movie_list[i]['id'], rearranged_movie_list[i]['predicted_rating']] for i in range(len(rearranged_movie_list))}
    #movie_cards = {"https://image.tmdb.org/t/p/original"+shuffled_movies[i]['poster_path'] if shuffled_movies[i]['poster_path'] else "/static/movie_placeholder.jpeg": [shuffled_movies[i]['title'],shuffled_movies[i]['original_title'],shuffled_movies[i]['vote_average'],datetime.strptime(shuffled_movies[i]['release_date'], '%Y-%m-%d').year if shuffled_movies[i]['release_date'] else "N/A", shuffled_movies[i]['id']] for i in range(len(shuffled_movies))}
    #movie_cards = {"https://image.tmdb.org/t/p/original"+movies_list[i]['poster_path'] if movies_list[i]['poster_path'] else "/static/movie_placeholder.jpeg": [movies_list[i]['title'],movies_list[i]['original_title'],movies_list[i]['vote_average'],datetime.strptime(movies_list[i]['release_date'], '%Y-%m-%d').year if movies_list[i]['release_date'] else "N/A", movies_list[i]['id']] for i in range(len(movies_list))}
    print("Search Results: -------------------------------- \n", movie_cards)
    print("\n--------------------------------------------------------------------")
    print("Run till here ....... \n")
    return render_template('recommend.html',movie_cards=movie_cards)

@app.route("/recommend",methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    rel_date = request.form['rel_date']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']
    rec_movies_org = request.form['rec_movies_org']
    rec_year = request.form['rec_year']
    rec_vote = request.form['rec_vote']
    rec_ids = request.form['rec_ids']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # Store the imdb_id in the session
    movie_id = request.form['movie_id']
    session['imdb_id'] = movie_id
    # Retrieve the imdb_id from the session
    movie_id = session.get('imdb_id')
    # Debug print
    print(f"Recommendation received: MOVIE ID - {movie_id}")

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies_org = convert_to_list(rec_movies_org)
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = convert_to_list_num(cast_ids)
    rec_vote = convert_to_list_num(rec_vote)
    rec_year = convert_to_list_num(rec_year)
    rec_ids = convert_to_list_num(rec_ids)
    
    # rendering the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')

    for i in range(len(cast_chars)):
        cast_chars[i] = cast_chars[i].replace(r'\n', '\n').replace(r'\"','\"') 
    
    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: [rec_movies[i],rec_movies_org[i],rec_vote[i],rec_year[i],rec_ids[i]] for i in range(len(rec_posters))}

    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    if(imdb_id != ""):
        # web scraping to get user reviews from IMDB site
        sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
        soup = bs.BeautifulSoup(sauce,'lxml')
        soup_result = soup.find_all("div",{"class":"text show-more__control"})

        reviews_list = [] # list of reviews
        reviews_status = [] # list of comments (good or bad)
        for reviews in soup_result:
            if reviews.string:
                reviews_list.append(reviews.string)
                # passing the review to our model
                movie_review_list = np.array([reviews.string])
                movie_vector = vectorizer.transform(movie_review_list)
                pred = clf.predict(movie_vector)
                reviews_status.append('Positive' if pred else 'Negative')

        # getting current date
        movie_rel_date = ""
        curr_date = ""
        if(rel_date):
            today = str(date.today())
            curr_date = datetime.strptime(today,'%Y-%m-%d')
            movie_rel_date = datetime.strptime(rel_date, '%Y-%m-%d')

        # combining reviews and comments into a dictionary
        movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}     

        # passing all the data to the html file
        return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
            vote_count=vote_count,release_date=release_date,movie_rel_date=movie_rel_date,curr_date=curr_date,runtime=runtime,status=status,genres=genres,movie_cards=movie_cards,reviews=movie_reviews,casts=casts,cast_details=cast_details)

    else:
        return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
            vote_count=vote_count,release_date=release_date,movie_rel_date="",curr_date="",runtime=runtime,status=status,genres=genres,movie_cards=movie_cards,reviews="",casts=casts,cast_details=cast_details)

@app.route('/store-rating', methods=['POST'])
def store_rating():
    data = request.json
    rating = data['rating']
    user_id = session['user_id']
    # Retrieve the imdb_id from the session
    movie_id = session.get('imdb_id')
    user = User.query.get(user_id)
    # Check if the movie ID exists in the ratings list
    ratings_dict = json.loads(user.ratings)

    if str(movie_id) in ratings_dict:
        ratings_dict[str(movie_id)] = str(rating)
    else:
        ratings_dict[str(movie_id)] = str(rating)

    user.ratings = json.dumps(ratings_dict)
    # Commit the changes to the database
    db.session.commit()
    print(f"Rating stored: Movie ID - {movie_id}, Rating - {rating}, User ID - {user_id}")
    return 'Rating stored successfully'

if __name__ == '__main__':
    # Create the Flask application context
    with app.app_context():
        # Create the database tables
        db.create_all()

        # Run the Flask application
        app.debug = True
        app.run(debug=True)