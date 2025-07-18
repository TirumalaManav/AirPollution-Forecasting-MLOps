
# Base image
FROM python:3.8-slim-buster

# Upgrade pip
RUN pip install --upgrade pip

# Copying all the files to the source directory
COPY . /src

# Setting the working directory
WORKDIR /src


# Make port 5000 available to the world outside this container
EXPOSE 5000

# Install required Python dependencies from the requirements file
RUN pip install -r requirements.txt

# Default command to run
CMD ["python", "docker_train.py"]

# # Builder stage
# FROM python:3.8-slim-buster as builder

# WORKDIR /app
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Final stage
# FROM python:3.8-slim-buster

# WORKDIR /app
# COPY --from=builder /opt/venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# COPY . .
# EXPOSE 5000
# CMD ["python", "docker_train.py"]
