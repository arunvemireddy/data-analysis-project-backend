from django import views
from django.urls import path,include
from .views import GetMovies, GetNotSeenMovies, GetRating, InsertDoc, LoginView, LogoutView, RecommendMovie,RegisterView, SearchUserMovie, SimilarUsers, UserMovies, UserView,SearchMovie

urlpatterns=[
    path('register/',RegisterView.as_view()),
    path('recommendmovie/',RecommendMovie.as_view()),
    path('searchmovie/',SearchMovie.as_view()),
    path('login/',LoginView.as_view()),
    path('user/',UserView.as_view()),
    path('logout/',LogoutView.as_view()),
    path('getMovies/',GetMovies.as_view()),
    # path('saveusermovies/',SaveUserMovieRating.as_view()),
    path('searchusermovie/',SearchUserMovie.as_view()),
    path('recommendusermovie/',SimilarUsers.as_view()),
    path('getRating/',GetRating.as_view()),
    path('userMov/',UserMovies.as_view()),
    path('notseen/',GetNotSeenMovies.as_view()),
    path('insertdoc/',InsertDoc.as_view())
]