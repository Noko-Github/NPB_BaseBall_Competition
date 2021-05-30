FROM jupyter/scipy-notebook

WORKDIR /home/work

RUN pip install lightgbm tqdm

EXPOSE 8888