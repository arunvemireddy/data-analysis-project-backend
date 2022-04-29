from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pymongo
from bson.json_util import dumps
from rest_framework.views import APIView
from api_app.models import User
from api_app.serializers import UserSerializer
from rest_framework.exceptions import AuthenticationFailed
import datetime
import jwt
import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise  import cosine_similarity
from django.core.paginator import Paginator
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import math
import re
import keras




my_client = pymongo.MongoClient('localhost:27017')
dbname = my_client['MDAP']
coll_name1 = dbname['movies']
coll_name2 = dbname['credits']
coll_name3 = dbname['user_movies']
coll_name4 = dbname['user_ratings']

# collection_name1 = dbname["loginDetails"]
# collection_name2 = dbname['movies']
# collection_name3 = dbname['ratings']
# coll_name3 = dbname['user_movies']
# med_details = collection_name1.find({})
# Create your views here.


# sample api call
@api_view()
def welcome(request):
    return Response({"message": "Welcome"})

# Register User API
class RegisterView(APIView):
    def post(self,request):
        serializer = UserSerializer(data=request.data)
        print(serializer)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

# Login User API
class LoginView(APIView):
    def post(self,request):
        email = request.data['email']
        password = request.data['password']
        user = User.objects.filter(email=email).first()
        if user is None:
            raise AuthenticationFailed('User not found!')
        if not user.check_password(password):
            raise AuthenticationFailed('Incorrect Password')

        payload={
            'id':user.id,
            'exp':datetime.datetime.utcnow()+datetime.timedelta(minutes=60),
            'iat':datetime.datetime.utcnow()
        }
        token = jwt.encode(payload,'secret',algorithm='HS256')
        response = Response()
        response.set_cookie(key='jwt',value=token,httponly=True)
        response.data={
            'jwt':token,
            'userId':user.id,
            'userName':user.name
        }
        return response

# Logout User API
class LogoutView(APIView):
    def post(self,request):
        response = Response()
        response.delete_cookie('jwt')
        response.data={
            'message':'success'
        }
        return response

# Test User API
class UserView(APIView):
    def get(self,request):
        token = request.COOKIES.get('jwt')
        print(token)
        if not token:
            raise AuthenticationFailed('unauthenticated')
        try:
            payload = jwt.decode(token, 'secret', algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            raise AuthenticationFailed('unauthenticated')

        user = User.objects.filter(id=payload['id']).first()
        serializer = UserSerializer(user)
        return Response(serializer.data)

#movies user did not watch
class GetNotSeenMovies(APIView):
    def post(self,request):
        print(type(request.data['userId']))
        token = request.headers.get('Authorization')
        if not token:
            raise AuthenticationFailed('unauthenticated')
        try:
            payload = jwt.decode(token, 'secret', algorithms=['HS256'])
            user = User.objects.filter(id=payload['id']).first()
            serializer = UserSerializer(user)
        except jwt.ExpiredSignatureError:
            raise AuthenticationFailed('unauthenticated')
        movies=coll_name3.find({},{"_id": False})
        ratings = coll_name4.find({'userId':request.data['userId']},{"_id": False,"rating":False,"timestamp":False,"userId":False})
        movies = list(movies)
        ratings = list(ratings)
        print(len(ratings))
        m_df = pd.DataFrame(movies)
        r_df = pd.DataFrame(ratings)
        r_df=r_df['movieId'].unique().tolist()
        for i in r_df:
            m_df.drop(m_df.index[m_df['movieId'] == i], inplace=True)
        res_df=m_df.head(100)
        res_list = res_df.to_dict('records')
        return Response({'message':res_list})

class InsertDoc(APIView):
    def post(self,request):
        dbname['user_ratings'].insert({"userId":request.data['userId'],"movieId":request.data['movieId'],"rating":request.data['rating']})
        return Response({'message':"inserted"})

# List Movies API
class GetMovies(APIView):
    def post(self,request):
        token = request.headers.get('Authorization')
        if not token:
            raise AuthenticationFailed('unauthenticated')
        try:
            payload = jwt.decode(token, 'secret', algorithms=['HS256'])
            user = User.objects.filter(id=payload['id']).first()
            serializer = UserSerializer(user)
        except jwt.ExpiredSignatureError:
            raise AuthenticationFailed('unauthenticated')
        movies=coll_name3.find({},{"_id": False})
        ratings = coll_name4.find({'userId':request.data['userId']},{"_id": False,"timestamp":False})
        movies = list(movies)
        ratings = list(ratings)
        m_df = pd.DataFrame(movies)
        r_df = pd.DataFrame(ratings)
        res_df = m_df.merge(r_df,on='movieId')
        res_df=res_df.sort_values(by="rating", ascending=False)
        res_list = res_df.to_dict('records')
        pages=math.ceil(len(res_list)/20)
        p = Paginator(res_list,20)
        page2 = p.page(request.data['page_number'])
        return Response({'message':res_list,'pages':pages})

# Search Movie API
class SearchMovie(APIView):
    def post(self,request):
        token = request.headers.get('Authorization')
        print(token)
        if not token:
            raise AuthenticationFailed('unauthenticated')
        try:
            payload = jwt.decode(token, 'secret', algorithms=['HS256'])
            user = User.objects.filter(id=payload['id']).first()
            serializer = UserSerializer(user)
            name=serializer.data['name']
        except jwt.ExpiredSignatureError:
            raise AuthenticationFailed('unauthenticated')
        msg = 'message'
        movie_name = request.data['moviename']
        # movies=coll_name1.find({'title':{'$regex':"^"+movie_name}},{"_id": False,'title':True})
        movies=coll_name1.find({'title':re.compile(movie_name,re.IGNORECASE)},{"_id": False,'title':True})
        movies = list(movies)
        return Response({msg:movies,'name':name})

# Recommend Movie API
class RecommendMovie(APIView):
    def post(self,request):
        token = request.headers.get('Authorization')
        print(token)
        if not token:
            raise AuthenticationFailed('unauthenticated')
        try:
            payload = jwt.decode(token, 'secret', algorithms=['HS256'])
            user = User.objects.filter(id=payload['id']).first()
            serializer = UserSerializer(user)
            name=serializer.data['name']
        except jwt.ExpiredSignatureError:
            raise AuthenticationFailed('unauthenticated')
        movies=coll_name1.find({},{"_id": False})
        credits=coll_name2.find({},{"_id": False})
        movie_list = list(movies)
        credit_list = list(credits)
        movies_df = pd.DataFrame(movie_list)
        credits_df = pd.DataFrame(credit_list)
        movies_df = movies_df.merge(credits_df, on="movie_id")
        print(credits_df)
        features = ["cast", "crew", "keywords", "genres"]
        for feature in features:
            movies_df[feature] = movies_df[feature].apply(literal_eval)
        movies_df['director'] = movies_df['crew'].apply(get_director)
        features = ['cast','keywords','genres']
        for feature in features:
            movies_df[feature] = movies_df[feature].apply(get_list)
        features = ['cast','keywords','director','genres']
        for feature in features:
            movies_df[feature] = movies_df[feature].apply(clean_data)
      
        movies_df['soup']=movies_df.apply(create_soup,axis=1)
      
        count_vectorizer = CountVectorizer(stop_words='english')
        count_matrix = count_vectorizer.fit_transform(movies_df['soup'])
        cosine_sim2 = cosine_similarity(count_matrix,count_matrix)
        movies_df = movies_df.reset_index()
        indices = pd.Series(movies_df.index,index=movies_df['title_y'])
        movie_name = request.data['moviename']
        res=get_recommendations(indices,cosine_sim2,movie_name,movies_df)
        return Response({'message':res,'name':name})


def get_recommendations(i,c,m,m_df):
    idx = i[m]
    similarity_scores = list(enumerate(c[idx]))
    similarity_scores= sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores= similarity_scores[1:11]
    movies_indices = [ind[0] for ind in similarity_scores]
    movies = m_df["title_y"].iloc[movies_indices]
    return movies

def get_director(x):
        for i in x:
            if i['job']=='Director':
                return i['name']
        return np.nan

def get_list(x):
    if isinstance(x,list):
        names = [i['name'] for i in x]
        if len(names)>3:
            names = names[:3]
        return names
    return []

def clean_data(row):
    if isinstance(row,list):
        return [str.lower(i.replace(" ",""))for i in row]
    else:
        if isinstance(row,str):
            return str.lower(row.replace(" ",""))
        else:
            return ""

def create_soup(features):
    return " ".join(features['keywords'])+ ' '+ ' '.join(features['cast'])+ ' '+features['director'] + ' ' + ' '.join(features['genres'])



class SearchUserMovie(APIView):
    def post(self,request):
        token = request.headers.get('Authorization')
        print(token)
        if not token:
            raise AuthenticationFailed('unauthenticated')
        try:
            payload = jwt.decode(token, 'secret', algorithms=['HS256'])
            user = User.objects.filter(id=payload['id']).first()
            serializer = UserSerializer(user)
            name=serializer.data['name']
        except jwt.ExpiredSignatureError:
            raise AuthenticationFailed('unauthenticated')
        msg = 'message'
        movie_name = request.data['movie_name']
        movies=coll_name3.find({'title':re.compile(movie_name,re.IGNORECASE)},{"_id": False,'title':True})
        movie = list(movies)
        return Response({msg:movie,'name':name})


class SimilarUsers(APIView):
    def post(self,request):
        token = request.headers.get('Authorization')
        print(token)
        if not token:
            raise AuthenticationFailed('unauthenticated')
        try:
            payload = jwt.decode(token, 'secret', algorithms=['HS256'])
            user = User.objects.filter(id=payload['id']).first()
            serializer = UserSerializer(user)
            name=serializer.data['name']
        except jwt.ExpiredSignatureError:
            raise AuthenticationFailed('unauthenticated')
        user_mov = coll_name3.find({},{"_id": False})
        user_rat = coll_name4.find({},{"_id": False})
        user_movies = list(user_mov)
        user_ratings = list(user_rat)
        user_movies=pd.DataFrame(user_movies)
        user_ratings=pd.DataFrame(user_ratings)
        final_dataset = user_ratings.pivot(index='movieId',columns='userId',values='rating')
        final_dataset.fillna(0,inplace=True)
        no_user_voted = user_ratings.groupby('movieId')['rating'].agg('count')
        no_movies_voted = user_ratings.groupby('userId')['rating'].agg('count')
        final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]
        final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]
        csr_data = csr_matrix(final_dataset.values,dtype=float)
        final_dataset.reset_index(inplace=True)
        knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        knn.fit(csr_data)
        x = get_movie_recommendation(request.data['movie_name'],user_movies,final_dataset,csr_data,knn)
        return Response({'message':x,'name':name})


def get_movie_recommendation(movie_name,user_movies,final_dataset,csr_data,knn):
    n_movies_to_reccomend = 10
    movie_list = user_movies[user_movies['title']==movie_name]  
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = user_movies[user_movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':user_movies.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    else:
        return "No movies found. Please check your input"

class GetRating(APIView):
    def post(self,request):
        user_rat = coll_name4.find({'movieId':request.data['movieId'],'userId':request.data['userId']},{"_id": False,"rating":True})
        rating=list(user_rat)
        return Response({'msg':rating})



def getMovieRecommendations(user_id,model):
    user_rat = coll_name4.find({},{"_id": False})
    df = pd.read_csv('user_ratings.csv')
    # doing preprocessing to encode movies and users
    user_ids = df["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}

    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    movie_ids = df["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

    df["user"] = df["userId"].map(user2user_encoded)
    df["movie"] = df["movieId"].map(movie2movie_encoded)

    user_mov = coll_name3.find({},{"_id": False})
    movie_df = pd.read_csv("user_movies.csv")

    movies_watched_by_user = df[df.userId == user_id]
    movies_not_watched = movie_df[~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
    ]["movieId"]
    movies_not_watched = list(
    set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
    )
    
    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
    user_encoder = userencoded2user.get(user_id)
    user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched))
    ratings = model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices]

    
    top_movies_user = (movies_watched_by_user.sort_values(by="rating", ascending=False).movieId.values)
    
    movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]

    for row in movie_df_rows.itertuples():
        recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
    high_rating = movie_df_rows.to_dict('records')
    recommended = recommended_movies.to_dict('records')
    return Response({'highrating':high_rating,'recommended':recommended})

class UserMovies(APIView):
    def post(self,request):
        x = keras.models.load_model('save_model/')
        user_id = request.data['user_id']
        return getMovieRecommendations(int(user_id),x)








     





    
