FROM alpine:3.7 as builder
RUN apk update && \
  apk --no-cache add \
    cmake \
    g++ \
    make \
    upx \
    linux-headers

ADD . /opt/sources
WORKDIR /opt/sources
RUN mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/tmp/build-dest .. && \
    make && make test && make install && upx -9 /tmp/build-dest/bin/opendlv-actuation


FROM alpine:3.7

WORKDIR /usr/bin
COPY --from=builder /tmp/build-dest/ /usr/
CMD ["/usr/bin/opendlv-actuation"]
