# --- Base image ---
FROM ubuntu:20.04

# --- Set environment for non-interactive apt installs ---
ENV DEBIAN_FRONTEND=noninteractive

# --- Install system packages ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      cmake libboost-all-dev libprotobuf-dev protobuf-compiler \
      clang git python3 python3-pip python3-venv build-essential libopencv-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Upgrade pip ---
RUN python3 -m pip install --upgrade pip

# --- Install Python packages using requirements.txt ---
COPY src/requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# --- Set working directory and copy project files ---
WORKDIR /workspace
COPY src/ /workspace/

# --- Default command ---
CMD ["bash"]
