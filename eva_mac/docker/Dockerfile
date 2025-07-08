# --- ベースイメージ ---
FROM ubuntu:20.04

# --- 共通依存パッケージのインストール ---
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      cmake libboost-all-dev libprotobuf-dev protobuf-compiler \
      clang git python3 python3-pip python3-venv build-essential && \
    rm -rf /var/lib/apt/lists/*

# --- Microsoft SEAL v3.6.0 のクローン & ビルド ---
RUN git clone -b v3.6.0 https://github.com/microsoft/SEAL.git /opt/SEAL && \
    cd /opt/SEAL && \
    cmake -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF . && \
    make -j$(nproc) && \
    make install

# --- EVA v1.0.0 のクローン & ビルド ---
RUN git clone -b v1.0.0 https://github.com/microsoft/EVA.git /opt/EVA && \
    cd /opt/EVA && \
    git submodule update --init && \
    cmake . && \
    make -j$(nproc)

# --- Python バインディングのインストール ---
RUN python3 -m pip install -U pip && \
    python3 -m pip install -e /opt/EVA/python/

# --- ワーキングディレクトリ & デフォルトコマンド ---
WORKDIR /workspace
CMD [ "bash" ]

