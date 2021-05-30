# How to use
```
# build image
docker build -t baseball:1.0.1 .

# run container
docker run -it --rm -p 8888:8888 --name basebass -v {abs path}:/home/work/ baseball:1.0.0
```