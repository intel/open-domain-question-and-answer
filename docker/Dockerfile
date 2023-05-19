FROM ubuntu:20.04

WORKDIR /home/user

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3  \
    python3-pip \
    libpoppler-cpp-dev \
    tesseract-ocr \
    wget \
    git \
    libtesseract-dev \
    poppler-utils \
    libmkl-dev

# Install PDF converter
RUN wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.04.tar.gz && \
    tar -xvf xpdf-tools-linux-4.04.tar.gz && cp xpdf-tools-linux-4.04/bin64/pdftotext /usr/local/bin

#clone the code repo
#RUN git clone --depth=1 https://github.com/intel/open-domain-question-and-answer.git
#WORKDIR /home/user/open-domain-question-and-answer
#RUN git submodule update --init --recursive

COPY / /home/user/open-domain-question-and-answer/
WORKDIR /home/user/open-domain-question-and-answer
# Install package
RUN pip install torch torchvision --force-reinstall --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip
# RUN pip install --no-cache-dir .[docstores,crawler,preprocessing,ocr,ray]
RUN pip install --no-cache-dir .[faiss,preprocessing,ocr,ray]
RUN pip install --no-cache-dir rest_api/
RUN pip install --no-cache-dir third_party/ColBERT/
RUN pip install numba
#RUN pip install faiss-1.6.3-py3-none-any.whl
RUN python3 -m pip install intel-extension-for-pytorch
RUN pip install intel-openmp
RUN ls /home/user
RUN pip freeze
RUN python3 -c "from haystack.utils.docker import cache_models;cache_models()"

# create folder for /file-upload API endpoint with write permissions, this might be adjusted depending on FILE_UPLOAD_PATH
RUN mkdir -p /home/user/open-domain-question-and-answer/rest_api/file-upload
RUN chmod 777 /home/user/open-domain-question-and-answer/rest_api/file-upload
RUN ln -s /usr/bin/python3.8 /usr/bin/python

EXPOSE 8000
ENV HAYSTACK_DOCKER_CONTAINER="HAYSTACK_CPU_CONTAINER"

# cmd for running the API
CMD ["gunicorn", "rest_api.application:app", "-b", "0.0.0.0", "-k", "uvicorn.workers.UvicornWorker", "--workers", "1", "--timeout", "180"]
