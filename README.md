# viva-stable-diffusion
## Before run the program, download the pre-trained model.
```
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
```

## [Builds Docker images from a Dockerfile](https://docs.docker.com/engine/reference/commandline/build/)

```
docker build -t <IMAGE_NAME>:<TAG> .
```

> eg: docker build -t viva-sd:latest .


## Create and start docker containers
Here, we are using docker compose to help us run the container with OPTIONS. 

```
docker compose up
```
