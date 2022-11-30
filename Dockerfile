FROM ubuntu:20.04

WORKDIR /home/user

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3  \
    python3-pip \
    libpoppler-cpp-dev \
    tesseract-ocr \
    wget \
    libtesseract-dev \
    poppler-utils

# Install PDF converter
RUN wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.04.tar.gz && \
    tar -xvf xpdf-tools-linux-4.04.tar.gz && cp xpdf-tools-linux-4.04/bin64/pdftotext /usr/local/bin

# Copy Haystack code
COPY haystack /home/user/haystack/
COPY third_party/ColBERT /home/user/ColBERT/
# Copy package files & models
COPY setup.py setup.cfg pyproject.toml VERSION.txt LICENSE README.md models* /home/user/
# Copy REST API code
COPY rest_api /home/user/rest_api/
#COPY faiss-index-so.json faiss-index-so.faiss /home/user/
# Install package
RUN pip install torch torchvision --force-reinstall --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip
RUN pip install --no-cache-dir .[docstores,crawler,preprocessing,ocr,ray]
RUN pip install --no-cache-dir rest_api/
RUN pip install --no-cache-dir ColBERT/
RUN pip install numba
RUN python3 -m pip install intel_extension_for_pytorch -f https://software.intel.com/ipex-whl-stable
RUN pip install intel-openmp
RUN ls /home/user
RUN pip freeze
RUN python3 -c "from haystack.utils.docker import cache_models;cache_models()"

# create folder for /file-upload API endpoint with write permissions, this might be adjusted depending on FILE_UPLOAD_PATH
RUN mkdir -p /home/user/rest_api/file-upload
RUN chmod 777 /home/user/rest_api/file-upload
RUN ln -s /usr/bin/python3.8 /usr/bin/python
# optional : copy sqlite db if needed for testing
#COPY qa.db /home/user/

# optional: copy data directory containing docs for ingestion
#COPY data /home/user/data

EXPOSE 8000
ENV HAYSTACK_DOCKER_CONTAINER="HAYSTACK_CPU_CONTAINER"

# cmd for running the API
CMD ["gunicorn", "rest_api.application:app", "-b", "0.0.0.0", "-k", "uvicorn.workers.UvicornWorker", "--workers", "1", "--timeout", "180"]
