from augmentation_functions import GpuAugmentation, MyAugmentation, ParallelAlbumentation
from PIL import Image
import time
import numpy as np
import gc
import albumentations as A
import matplotlib.pyplot as plt
import cupy as cp


if __name__ == '__main__':
    width = # images width
    height = # images height
    channels = # images channels

    n_images = # number of images to augment
    batch_size = # number of images in a single batch
    transform = A.ToGray(p=1) 

    images = # load your images
    print('Ready for the augmentation')

    # COLLECT TIMES FOR SEQUENTIAL EXECUTION
    ####################################################################################################################
    start = time.time()
    new_from_album = [transform(image=images[i])['image'] for i in range(len(images))]
    print('Number of images: ', len(new_from_album))
    stop = time.time()
    seq_time = stop - start
    print(f"Execution time Single Albumentation: {seq_time} s")

    del new_from_album
    gc.collect()

    # COLLECT TIMES FOR ParallelAlbumentation EXECUTION
    ####################################################################################################################
    parallel_albumentation = ParallelAlbumentation(images, batch_size)
    start = time.time()
    new_images = parallel_albumentation.brightness(.4)
    stop = time.time()

    # ParallelAlbumentation.show_image(new_images, 1)
    print(f"Execution time Parallel Albumentation: {stop - start} s   -----> SpeedUp: {seq_time / (stop - start)} s")

    del parallel_albumentation
    del new_images
    gc.collect()

    # COLLECT TIMES FOR CustomAugmentation EXECUTION
    ####################################################################################################################
    
    augmentation = MyAugmentation(images, width, height, channels, batch_size)
    start = time.perf_counter()
    augmentation.crop_and_resize(60, 60)
    stop = time.perf_counter()
    print(f"Execution time MyAugmentation: {stop - start} s   -----> SpeedUp: {seq_time / (stop - start)} s")
    del augmentation
    gc.collect()
