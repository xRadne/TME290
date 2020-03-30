FROM alpine:3.7 as builder
RUN apk update && \
    apk --no-cache add \
        ca-certificates \
        cmake \
        g++ \
        make \
        linux-headers
RUN apk add libcluon --no-cache --repository \
    https://chrberger.github.io/libcluon/alpine/v3.7 --allow-untrusted
ADD . /opt/sources
WORKDIR /opt/sources
RUN mkdir /tmp/build && \
    cd /tmp/build && \
    cmake -D CMAKE_BUILD_TYPE=Release /opt/sources && \
    make && make test && cp helloworld /tmp

# Deploy
FROM alpine:3.7
RUN apk update && \
    apk --no-cache add \
        libstdc++
RUN mkdir /opt
WORKDIR /opt
COPY --from=builder /tmp/helloworld .
CMD ["/opt/helloworld"]
