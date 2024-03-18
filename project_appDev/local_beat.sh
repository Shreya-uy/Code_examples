#! /bin/sh
echo "======================================================================"
echo "Celery beat installation"
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

celery -A main.celery beat --max-interval 1 -l info

deactivate
