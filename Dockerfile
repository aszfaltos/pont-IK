FROM huggingface/transformers-all-latest-gpu
LABEL authors="aszfalt"

# Better init
ENV TINI_VERSION="v0.19.0"

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

# add local scripts to path
ENV PATH /home/user/.local/bin:$PATH

# set workdir
WORKDIR /project

# non-root user
RUN useradd -m -r user && \
    chown user /project

# set user
USER user

# Update essentials
RUN pip install --upgrade pip setuptools wheel

# install requirements
COPY requirements_docker.txt .
RUN pip install -r requirements_docker.txt

# copy project
COPY . .

ENV PYTHONPATH /project:$PYTHONPATH

# set up git hash for versioning
ARG GIT_HASH
ENV GIT_HASH=${GIT_HASH:-dev}

# set entry point
ENTRYPOINT ["/tini", "--"]

# set location and privilidge for docker