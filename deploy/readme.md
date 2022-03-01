## Use C code to load the images

You can use the C code in `./C/` to load the images from the SD card with multi-process.

```bash
g++ -shared -O2 load_image.cpp -o load_image.so -fPIC `pkg-config opencv --cflags --libs` -lpthread
```

## Evaluate the accuracy

You can use the notebook in `./test/` to evaluate the accuracy of the model.