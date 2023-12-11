FROM python:3.10

WORKDIR /app

COPY ./requirements.txt /app

RUN apt-get update && apt-get install -y \
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
