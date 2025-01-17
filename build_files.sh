#!/bin/bash
# upgrade pip
python3.9 -m ensurepip
python3.9 -m pip install --upgrade pip

echo "Building the project..."
python3.9 -m pip install -r requirements.txt

# Collect static files
# python manage.py collectstatic --noinput

# echo "Make Migration..."
# python3.9 manage.py makemigrations --noinput
# python3.9 manage.py migrate --noinput

echo "Collect Static..."
python3.9 manage.py collectstatic --noinput --clear