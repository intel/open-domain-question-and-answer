FROM ubuntu:20.04

# RUN apt-get update && apt-get install -y curl git pkg-config cmake

# copy code
COPY . /ui

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3  \
    python3-pip

# install as a package
RUN pip install --upgrade pip && \
    pip install /ui/ \
    pip install pyyaml

RUN ln -s /usr/bin/python3.8 /usr/bin/python
WORKDIR /ui
EXPOSE 8501

# cmd for running the API
CMD ["python", "-m", "streamlit", "run", "ui/webapp.py"]
