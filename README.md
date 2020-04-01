# Coding Playground #

This repository contains random code for things that I learn about or experiment with.

<p align="center">
  <img src="https://vignette.wikia.nocookie.net/dexterslab/images/7/70/Dex_dexter_174x252.png/revision/latest?cb=20150331204700">
</p>

## CUDA

My implementation of the examples from [CUDA by Example: An Introduction to General-Purpose GPU Programming](https://www.amazon.com/CUDA-Example-Introduction-General-Purpose-Programming/dp/0131387685/ref=pd_bbs_sr_1/103-9839083-1501412?ie=UTF8&s=books&qid=1186428068&sr=1-1).

#### Building

From the source directory, run the following:

```
nvcc INPUT_FILE.cu -o EXECUTABLE_NAME
```

If you get a linking error, make sure to have OpenGL and GLUT installed on your computer. Then compile with the `-lGL -lglut` flags.
