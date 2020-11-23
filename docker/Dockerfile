FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update
RUN apt-get install -y libsndfile1
RUN apt-get install -y ffmpeg
RUN pip3 install -U pip
RUN pip3 install numba==0.48
RUN pip3 install pandas
RUN pip3 install -U efficientnet
RUN pip3 install librosa
RUN pip3 install pillow
RUN pip3 install -U scikit-learn
RUN pip3 install tqdm
RUN pip3 install keras
RUN pip3 install jupyter

RUN mkdir /project
WORKDIR /project
RUN mkdir data

EXPOSE 8888
CMD jupyter notebook  --no-browser --ip=0.0.0.0 --allow-root