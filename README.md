Back end
step 1 - create environment
python -m venv env

step 2 - activate environment
cd env/Scripts 
then ./activate

step 3 - install packages
pip install pymongo==3.12.1
pip install pyjwt
pip install jwt
pip install tensorflow
pip install keras
pip install sklearn
pip install pandas
pip install djongo
pip install djongoframework
pip install django-cors-headers
pip install django

Note - for pymongo use 3.12.1 version only

step 4 : make migrations
python manage.py makemigrations

step 5 : migrate
python manage.py migrate

step 6 : run project
python manage.py runserver

Frond end
step 1 - install packages

Database
step 1 - load database with movies.csv,credits.csv, user_ratings.csv,user_movies.csv



