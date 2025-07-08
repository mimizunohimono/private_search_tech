# private_search_tech
This repo stores some source code which realize PIR search.

## How to use

### Setting for EVA (Homomorphic Encryption Library)

```bash
$ sudo apt install cmake libboost-all-dev libprotobuf-dev protobuf-compiler build-essential g++ libstdc++-12-dev
$ sudo apt install clang
$ sudo update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100
$ sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100
$ git clone -b v3.6.4 https://github.com/microsoft/SEAL.git
$ cd SEAL
$ cmake -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF .
$ make -j
$ sudo make install

```

