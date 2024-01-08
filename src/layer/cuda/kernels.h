#ifndef SRC_LAYER_KERNELS_KERNELS_H_
#define SRC_LAYER_KERNELS_KERNELS_H_
#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include "utils.h"

class kernel_manager
{
public:
    void get_device_info();
    void conv_forward(const float* in, float* out, const float* weight, const int n_samples, 
        const int channel_in, const int channel_out, 
        const int height_in, const int width_in, const int kernel_width,
        const int kernel_type);
};

#endif // SRC_LAYER_KERNELS_KERNELS_H_