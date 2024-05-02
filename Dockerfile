FROM python:3.10-bookworm
LABEL authors="aszfalt"

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

# set up gradio
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# set entrypoint
ENTRYPOINT ["bash"]