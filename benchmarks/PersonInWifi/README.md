## Build OpenPose

### Download and create virtual environment
```shell!
$ git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
$ virtualenv openpose
```

### Prequisite
```shell!
$ pip install opencv-python # opencv
```

### enter environment
```shell!
$ cd openpose
$ source bin/activate
```

### cmake
```shell!
$ mkdir build
$ cd build
$ cmake -DBUILD_PYTHON=ON ..
```
> Ignore any download model error and cmake again until success.

### make
```shell!
$ make -j4
```
And you will see `pyopenpose.cpython-310-x86_64-linux-gnu.so` in `openpose/build/python/openpose`.

- copy `.so` to `openpose/lib/python3.10/site-packages`

### Download model weight
Download model weight from [Google Drive](https://drive.google.com/file/d/1QCSxJZpnWvM00hx49CJ2zky7PWGzpcEh/edit) and extract it.
Move body_25 weight to `openpose/models/pose/body_25`