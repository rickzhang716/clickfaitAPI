FROM python:3.9.7
COPY ./clickbaitEval2.py /deploy/
COPY ./requirements.txt /deploy/
ADD ./tfmodels2 /deploy/
ADD tfmodels2 /deploy/tfmodels2/
WORKDIR /deploy/
RUN pip install -r ./requirements.txt
EXPOSE 80
ENTRYPOINT [ "python3","clickbaitEval2.py"]

