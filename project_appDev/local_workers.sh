#! /bin/sh
echo "======================================================================"
echo " This will setup the celery workers in a virtual environment"
echo "----------------------------------------------------------------------"
if [ -d ".env" ];
then
    echo "Enabling virtual env"
else
    echo "No Virtual env. Please run setup.sh first"
    exit N
fi

# Activate virtual env
. .env/bin/activate

celery -A main.celery worker -l info

deactivate
