# ==================== Building Model Layer ===========================
# This is a little trick to improve caching and minimize rebuild time
# and bandwidth. Note that RUN commands only cache-miss if the prior layers
# miss, or the dockerfile changes prior to this step.
# To update these patch files, be sure to run build with --no-cache
FROM alpine as model_data
RUN apk --no-cache --update-cache add wget
WORKDIR /data/patch_experts

RUN wget -q https://www.dropbox.com/s/7na5qsjzz8yfoer/cen_patches_0.25_of.dat &&\
    wget -q https://www.dropbox.com/s/k7bj804cyiu474t/cen_patches_0.35_of.dat &&\
    wget -q https://www.dropbox.com/s/ixt4vkbmxgab1iu/cen_patches_0.50_of.dat &&\
    wget -q https://www.dropbox.com/s/2t5t1sdpshzfhpj/cen_patches_1.00_of.dat

## ==================== Install Ubuntu Base libs ===========================
## This will be our base image for OpenFace, and also the base for the compiler
## image. We only need packages which are linked

FROM ubuntu:18.04 as ubuntu_base

LABEL maintainer="Michael McDermott <mikemcdermott23@gmail.com>"

ARG DEBIAN_FRONTEND=noninteractive

# todo: minimize this even more
RUN apt-get update -qq &&\
    apt-get install -qq curl &&\
    apt-get install -qq --no-install-recommends \
        libopenblas-dev liblapack-dev \
        libavcodec-dev libavformat-dev libswscale-dev \
        libtbb2 libtbb-dev libjpeg-dev \
        libpng-dev libtiff-dev \
        openssl libssl-dev &&\
    rm -rf /var/lib/apt/lists/*

## ==================== Build-time dependency libs ======================
## This will build and install opencv and dlib into an additional dummy
## directory, /root/diff, so we can later copy in these artifacts,
## minimizing docker layer size
## Protip: ninja is faster than `make -j` and less likely to lock up system
FROM ubuntu_base as cv_deps

WORKDIR /root/build-dep
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -qq -y \
        cmake ninja-build pkg-config build-essential checkinstall\
        g++-8 &&\
    rm -rf /var/lib/apt/lists/* &&\
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8


RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Descarga, compila e instala CMake v3.12.4 (o la versión deseada)
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.12.4/cmake-3.12.4.tar.gz && \
    tar xf cmake-3.12.4.tar.gz && \
    cd cmake-3.12.4 && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf cmake-3.12.4 cmake-3.12.4.tar.gz

##        llvm clang-3.7 libc++-dev libc++abi-dev  \
## ==================== Building dlib ===========================

RUN curl http://dlib.net/files/dlib-19.13.tar.bz2 -LO &&\
    tar xf dlib-19.13.tar.bz2 && \
    rm dlib-19.13.tar.bz2 &&\
    mv dlib-19.13 dlib &&\
    mkdir -p dlib/build &&\
    cd dlib/build &&\
    cmake -DCMAKE_BUILD_TYPE=Release -G Ninja .. &&\
    ninja && \
    ninja install && \
    DESTDIR=/root/diff ninja install &&\
    ldconfig

## ==================== Building OpenCV ======================
ENV OPENCV_VERSION=4.1.0

RUN curl https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz -LO &&\
    tar xf ${OPENCV_VERSION}.tar.gz && \
    rm ${OPENCV_VERSION}.tar.gz &&\
    mv opencv-${OPENCV_VERSION} opencv && \
    mkdir -p opencv/build && \
    cd opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D WITH_TBB=ON -D WITH_CUDA=OFF \
        -DWITH_QT=OFF -DWITH_GTK=OFF\
        -G Ninja .. && \
    ninja && \
    ninja install &&\
    DESTDIR=/root/diff ninja install

## ==================== Building RabbitMQ ======================

RUN curl https://github.com/alanxz/rabbitmq-c/archive/refs/tags/v0.13.0.tar.gz -LO &&\
    tar xf v0.13.0.tar.gz && \
    rm v0.13.0.tar.gz &&\
    mv rabbitmq-c-0.13.0 rabbitmq && \
    mkdir -p rabbitmq/build && \
    cd rabbitmq/build && \
    cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
        -G Ninja .. && \
    ninja && \
    ninja install &&\
    DESTDIR=/root/diff ninja install &&\
    ldconfig

RUN apt-get update -qq && \
    apt-get install -qq --no-install-recommends \
        libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://github.com/alanxz/SimpleAmqpClient/archive/refs/tags/v2.5.1.tar.gz -LO &&\
    tar xf v2.5.1.tar.gz && \
    rm v2.5.1.tar.gz &&\
    mv SimpleAmqpClient-2.5.1 SimpleAmqpClient && \
    mkdir -p SimpleAmqpClient/build && \
    cd SimpleAmqpClient/build && \
    cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
        -G Ninja .. && \
    ninja && \
    ninja install &&\
    DESTDIR=/root/diff ninja install &&\
    ldconfig

RUN curl https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz -LO &&\
    tar xf v3.11.3.tar.gz && \
    rm v3.11.3.tar.gz &&\
    mv json-3.11.3 json && \
    mkdir -p json/build && \
    cd json/build && \
    cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
        -G Ninja .. && \
    ninja && \
    ninja install &&\
    DESTDIR=/root/diff ninja install &&\
    ldconfig

## ==================== Building OpenFace ===========================
FROM cv_deps as openface
WORKDIR /root/openface

COPY ./ ./

COPY --from=model_data /data/patch_experts/* \
    /root/openface/lib/local/LandmarkDetector/model/patch_experts/

RUN mkdir -p build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -G Ninja .. && \
    ninja &&\
    DESTDIR=/root/diff ninja install


## ==================== Streamline container ===========================
## Clean up - start fresh and only copy in necessary stuff
## This shrinks the image from ~8 GB to ~1.6 GB
# FROM ubuntu_base as final

# WORKDIR /root

# # Copy in only necessary libraries
# COPY --from=openface /root/diff /

# # Since we "imported" the build artifacts, we need to reconfigure ld
# RUN ldconfig
