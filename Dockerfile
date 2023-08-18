FROM python:3.8

WORKDIR /app

# Create the environment:
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY remove_baseline.py utils.py ./

# The code to run when container is started:
CMD ["python", "remove_baseline.py",   \
     "-i", "/datadir", "-d", "/datadir", \
     "--base-line", "100", "--mode", "train"]
