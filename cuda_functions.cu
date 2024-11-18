
// FLIP
########################################################################################################################
__global__ void flip(unsigned char* input, unsigned char* output, int width,
                                             int height, int channels, int batch_size, int axis) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int i = 0; i < batch_size; i++) {
            if (axis == 0) {
                for (int c = 0; c < channels; ++c) {
                    output[i * width * height * channels + y * width * channels + (width - 1 - x) * channels + c] =
                        input[i * width * height * channels + y * width * channels + x * channels + c];
                }
            } else {
                for (int c = 0; c < channels; ++c) {
                    output[i * width * height * channels + (height - 1 - y) * width * channels + x * channels + c] =
                        input[i * width * height * channels + y * width * channels + x * channels + c];
                }
            }
        }
    }
}


// TO GREY
########################################################################################################################
__global__ void to_grey(unsigned char* input, unsigned char* output, int width,
                                            int height, int input_channels, int batch_size) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_dim = width * height * input_channels;
    int b_pos;
    unsigned char gray_level;


    if (x < width && y < height) {
        for (int i = 0; i < batch_size; i++){
            b_pos = i * batch_dim;
            gray_level = (unsigned char) (
                        0.299 * input[b_pos + y * width * input_channels + x * input_channels]  +
                        0.587 * input[b_pos + y * width * input_channels + x * input_channels + 1] +
                        0.114 * input[b_pos + y * width * input_channels + x * input_channels + 2]
                    );

            output[b_pos + y * width + x] = gray_level;
        }
    }
}

// CHANNEL SHUFFLE
########################################################################################################################
__global__ void channel_shuffle(unsigned char* input, unsigned char* output, int width,
                                            int height, int channels, int batch_size) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned char first, second, third;

    if (x < width && y < height) {
        for (int i = 0; i< batch_size; i++) {
            first = input[i * width * height * channels + y * width * channels + x * channels];
            second = input[i * width * height * channels + y * width * channels + x * channels + 1];
            third = input[i * width * height * channels + y * width * channels + x * channels + 2];

            output[i * width * height * channels + y * width * channels + x * channels] = second;
            output[i * width * height * channels + y * width * channels + x * channels + 1] = third;
            output[i * width * height * channels + y * width * channels + x * channels + 2] = first;
        }
    }
}

// CROP
########################################################################################################################
__global__ void crop(unsigned char* input, unsigned char* output, int width,
                                        int height, int channels, int batch_size, int x_vertex, int y_vertex,int crop_width,
                                        int crop_height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < crop_width && y < crop_height) {
        int input_x = x + x_vertex;
        int input_y = y + y_vertex;

        for (int i = 0; i < batch_size; i++) {
            for (int c = 0; c < channels; ++c) {
                output[i * width * height * channels + y * width * channels + x * channels + c] =
                input[i * width * height * channels + input_y * width * channels + input_x * channels + c];
            }
        }
    }
}

// RESIZE
########################################################################################################################
__global__ void resize(unsigned char* input, unsigned char* output, int width,
                              int height, int channels, int batch_size,  int crop_width, int crop_height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float x_scale = (float)crop_width / width;
        float y_scale = (float)crop_height / height;

        float origin_x = x * x_scale;
        float origin_y = y * y_scale;

        int x1 = (int)floor(origin_x);
        int x2 = min(x1 + 1, crop_width - 1);

        int y1 = (int)floor(origin_y);
        int y2 = min(y1 + 1, crop_height - 1);

        float w1 = (x2 - origin_x) * (y2 - origin_y);
        float w2 = (origin_x - x1) * (y2 - origin_y);
        float w3 = (x2 - origin_x) * (origin_y - y1);
        float w4 = (origin_x - x1) * (origin_y - y1);

        for (int i = 0; i < batch_size; i++) {
            for (int c = 0; c < channels; ++c) {
                float pixel_value =
                    (w1 * input[i * width * height * channels + y1 * width * channels + x1 * channels + c] +
                     w2 * input[i * width * height * channels + y1 * width * channels + x2 * channels + c] +
                     w3 * input[i * width * height * channels + y2 * width * channels + x1 * channels + c] +
                     w4 * input[i * width * height * channels + y2 * width * channels + x2 * channels + c]);

                output[i * width * height * channels + y * width * channels + x * channels + c] = (unsigned char)fminf(fmaxf(pixel_value, 0.0f), 255.0f);

            }
        }
    }
}

// OCCLUSION
########################################################################################################################
__global__ void occlusion(unsigned char* input, unsigned char* output, int width,
                                        int height, int channels, int batch_size, int x_lower, int x_upper, int y_lower, int y_upper) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int x_b, y_b;

    if (x < width && y < height) {
        for (int i = 0; i < batch_size; i++) {
            x_b = (int) ((x_lower + width) / (x + width)) + (int) ((x + width) / (x_upper + width));
            y_b = (int) ((y_lower + height) / (y + height)) + (int) ((y + height) / (y_upper + height));

            for (int c = 0; c < channels; ++c) {
                output[i * width * height * channels + y * width * channels + x * channels + c] =
                input[i * width * height * channels + y * width * channels + x * channels + c] * (int) ((x_b + y_b + 1) / 2);
            }
        }
    }
}

// CONTRAST
########################################################################################################################
__global__ void contrast(unsigned char* input, unsigned char* output, int width,
                                             int height, int channels, int batch_size, unsigned char* means, int brightness) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int check;

    float pixel_value;
    float contrast_factor = (float)brightness / 10;

    if (x < width && y < height) {
        for (int i = 0; i < batch_size; i++) {
            for (int c = 0; c < channels; ++c) {
                pixel_value = (float) input[i * width * height * channels + y * width * channels + x * channels + c];
                pixel_value = (pixel_value - (float) means[c]) * contrast_factor + (float) means[c];
                check = (int) ((pixel_value + 510) / 765);

                output[i * width * height * channels + y * width * channels + x * channels + c] = (unsigned char) (pixel_value * (1 - check)
                                                                    + 255 * check);
            }
        }
    }
}

// BRIGHTNESS
########################################################################################################################
__global__ void brightness(unsigned char* input, unsigned char* output, int width,
                                             int height, int channels, int batch_size, int percentage, int to_add) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned char delta, pixel_value, division;
    unsigned char percent_pixel = (unsigned char)((percentage * 255) / 100);


    if (x < width && y < height) {
        for (int i = 0; i< batch_size; i++) {
            if (to_add == 0) {
                for (int c = 0; c < channels; ++c) {
                    pixel_value = input[i * width * height * channels + y * width * channels + x * channels + c];
                    delta = (unsigned char)(255) - pixel_value;
                    division = (unsigned char) ((pixel_value + percent_pixel) / 255);
                    output[i * width * height * channels + y * width * channels + x * channels + c] = pixel_value +
                                                            (1 - division) * percent_pixel + division * delta;
                }
            } else {
                for (int c = 0; c < channels; ++c) {
                    pixel_value = input[i * width * height * channels + y * width * channels + x * channels + c];
                    division = (unsigned char) ((pixel_value + 255) / (percent_pixel + 255));
                    output[i * width * height * channels + y * width * channels + x * channels + c] = division * (pixel_value - percent_pixel);
                }
            }
        }
    }
}
