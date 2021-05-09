FROM jupyter/base-notebook:ubuntu-20.04
WORKDIR /usr/src/code
COPY data.zip data.zip
COPY "node2vec+xgboost.ipynb" notebook.ipynb

USER root
RUN apt update
RUN apt install -y unzip gcc g++
RUN unzip data.zip data/*
RUN pip install --upgrade gensim
RUN pip install jupyter jupyterlab xgboost
RUN pip install networkx numpy pandas matplotlib scikit-learn
ENV JUPYTER_ENABLE_LAB=yes

RUN chmod 777 .
RUN chmod 777 ./notebook.ipynb
#RUN jupyter lab --allow-root --port 8324
EXPOSE 8888
# not to use root to run
USER 1000
CMD ["jupyter", "lab"]

