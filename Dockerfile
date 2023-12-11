FROM nvidia/cuda:11.8.0-base-ubuntu22.04

WORKDIR /app

COPY ./requirements.txt /app

RUN apt-get update && apt-get install -y \
    pip \
    ca-certificates \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt

# Can cache until here if no changes in requirements -> Skip reinstalling requirements on every build
COPY . /app

# Expose the ports that your app uses
EXPOSE 80

# Add a command to run your application
CMD ["bash", "-c", "uvicorn agent.api:app --host 0.0.0.0 --port 80"]
