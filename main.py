from augmentation_functions import GpuAugmentation, MyAugmentation, ParallelAlbumentation
from PIL import Image
import time
import numpy as np
import gc
import albumentations as A
import matplotlib.pyplot as plt

import cupy as cp


def return_images_imagenet(max_n):
    all_images = np.load("D:/Download/images.npz")
    i = 0
    imgs = []

    for key in all_images:
        imgs.append(all_images[key])
        i += 1
        if i == max_n:
            break

    del all_images
    gc.collect()

    imgs = np.array(imgs)

    return imgs


def collect_times(images, batch_size):
    width, height, channels = 500, 375, 3
    transform = A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1),
                                           contrast_limit=(0, 0),
                                           p=1)
    n_images = len(images)

    # print(n_images, batch_size)

    start = time.time()
    new_images = [transform(image=images[i])['image'] for i in range(n_images)]
    stop = time.time()
    seq_time = stop - start

    del new_images
    gc.collect()

    parallel_albumentation = ParallelAlbumentation(images, 128)
    start = time.time()
    new_images = parallel_albumentation.brightness(0.1)
    stop = time.time()
    par_time = stop - start

    del new_images
    gc.collect()

    augmentation = MyAugmentation(images, width, height, channels, int(n_images / 1))
    start = time.perf_counter()
    augmentation.brightness(10)
    stop = time.perf_counter()
    aug_time = stop - start

    # augmentation.from_batch_to_image()
    # augmentation.show_image(n_images - 1)

    del augmentation
    cp.cuda.Stream().synchronize()
    cp.get_default_memory_pool().free_all_blocks()

    gc.collect()

    # print(seq_time, par_time, aug_time)
    return seq_time, par_time, aug_time


def print_times(t, y, z, x):
    plt.plot(x, t, label='Sequential Albumentation', color='blue')
    plt.plot(x, y, label='Parallel Albumentation', color='green')
    plt.plot(x, z, label='Custom Augmentation', color='red')

    plt.xlabel('Number of images')
    plt.ylabel('Times')
    plt.title('Times curves')

    plt.yscale("log")

    plt.legend()

    plt.show()

def print_speedup1(y, x):
    plt.plot(x, y, label='Parallel Albumentation', color='green')

    plt.xlabel('Number of images')
    plt.ylabel('Speedup')
    plt.title('Speedup - Albumentation')
    # plt.yscale("log")

    # plt.legend()

    plt.show()

def print_speedup2(y, x):
    plt.plot(x, y, label='Custom Augmentation', color='red')

    plt.xlabel('Number of images')
    plt.ylabel('Speedup')
    plt.title('Speedup - Custom function')
    # plt.yscale("log")

    # plt.legend()

    plt.show()


def warm_up(images, width, height, channels, batch_size):
    augmentation = MyAugmentation(images, width, height, channels, batch_size)
    augmentation.brightness(10)
    del augmentation
    cp.cuda.Stream().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()


if __name__ == '__main__':
    # data_directory = 'dataset/cifar-10-batches-py/'
    # images, labels = alf.load_cifar10(data_directory)

    '''image = GpuAugmentation(n_new_images=1, image_path='images/cat.jpg')
        image.channel_shuffle(0)
        image = image.from_array_to_image()
        image[0].show()'''

    '''# image = GpuAugmentation(n_new_images=1, image_path='images/cat.jpg')
    # print(benchmark(image.rotation, (0, 270), n_repeat=50))

    image_obj = []

    for image in images:
        image_obj.append(GpuAugmentation(n_new_images=1, image_matrix=image, width=32, height=32, channels=3))

    start = time.time()

    new_images = [image_obj[i].reflection(0) for i in range(len(images))]
    stop = time.time()

    print(f"Execution time: {stop - start} s")

    start = time.time()
    new_from_album = [alf.a_reflection(images[i]) for i in range(len(images))]
    # print(benchmark(alf.a_reflection, (images, ), n_repeat=1))
    stop = time.time()

    print(f"Execution time Albumentation: {stop - start} s")

    start = time.time()
    # new_images = alf.parallelize_function(images)
    # print(benchmark(alf.batch_process, (images, 1000), n_repeat=10))
    new_images = alf.batch_process(images, 1000)
    stop = time.time()
    print(f"Execution time Parallel Albumentation: {stop - start} s")'''

    width = 500
    height = 375
    channels = 3

    n_images = 700
    batch_size = 600
    transform = A.ToGray(p=1)

    images = return_images_imagenet(n_images)
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
    warm_up(images, width, height, channels, batch_size)
    
    augmentation = MyAugmentation(images, width, height, channels, batch_size)
    start = time.perf_counter()
    augmentation.crop_and_resize(60, 60)
    stop = time.perf_counter()
    print(f"Execution time MyAugmentation: {stop - start} s   -----> SpeedUp: {seq_time / (stop - start)} s")
    del augmentation
    gc.collect()


    '''max_images = 1280
    images = return_images_imagenet(max_images)
    sizes = [32, 64, 128, 256, 512, 1024, 1536, 2048]

    n_iteration = 10
    image_dim = []
    seq_times = []
    par_times = []
    aug_times = []

    par_s = []
    aug_s = []

    # dim = int(max_images / n_iteration)
    dim = 128

    width, height, channels = 500, 375, 3
    warm_up(images[:dim], width, height, channels, dim)

    for i in range(1, n_iteration + 1):
        n_images = dim * i
        # n_images = max_images
        batch_size = int(n_images / 8)
        # batch_size = sizes[i - 1]

        x, y, z = collect_times(images[:n_images], batch_size)

        image_dim.append(n_images)
        seq_times.append(x)
        par_times.append(y)
        aug_times.append(z)

        par_s.append(x / y)
        aug_s.append(x / z)

        print(f'Iteration {i} completed.')

    print_times(seq_times, par_times, aug_times, image_dim)
    print_speedup1(par_s, image_dim)
    print_speedup2(aug_s, image_dim)'''


