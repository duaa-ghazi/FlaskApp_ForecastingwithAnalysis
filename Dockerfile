FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install requests
RUN pip3 install pandas

COPY . .

CMD [ "python3", "app.py"]