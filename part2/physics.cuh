#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 512

extern double size;
//
//  benchmarking program
//

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor);

__global__ void compute_forces_gpu(particle_t * particles, int n);

__global__ void move_gpu (particle_t * particles, int n, double size);