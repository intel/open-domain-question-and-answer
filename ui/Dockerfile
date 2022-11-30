FROM ubuntu:20.04

WORKDIR /home/user

RUN apt-get update && apt-get install -y \
    python3  \
    python3-pip

# copy code
RUN mkdir ui/
COPY setup.py /home/user/ui
COPY utils.py /home/user/ui
COPY webapp.py /home/user/ui
COPY eval_labels_example.csv /home/user/

# install as a package
RUN pip install --upgrade pip
RUN pip install ui/
RUN ln -s /usr/bin/python3.8 /usr/bin/python

EXPOSE 8501

# cmd for running the API
CMD ["python", "-m", "streamlit", "run", "ui/webapp.py"]
