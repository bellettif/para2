/*
 *  quadtree.h
 *  Header file for a quadtree.
 *
 *  https://github.com/ninjin/barnes-hut-sne/blob/master/quadtree.h
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 */


#pragma once

#include <cuda.h>

#include "physics.cuh"
#include "common.h"

#define density 0.0005

typedef struct {
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    particle_t* global_particles;
    int n_global_particles;
    particle_t* shared_particles;
    int n_shared_particles;
} region;


__global__ void apply_forces(region *r, const int offset){

    int i = threadIdx.x;

    if(i >= r->n_shared_particles - offset) return;

    particle_t& part = *(r->shared_particles + offset + i);
    part.ax = 0;
    part.ay = 0;

    for(int j = 0; j < r->n_shared_particles; ++j){
        apply_force(part, r->shared_particles[j]);
    }

}

__global__ void move_particles(region *r, const int offset){

    int i = threadIdx.x;

    if(i >= r->n_shared_particles - offset) return;

    move(*(r->shared_particles + offset + i));

}


__global__ void simulate(region* r, const int stride,
                         const int n_r_x, int const n_r_y,
                         const int N_BLOCKS,
                         const int N_STEPS){

    int block_idx = blockIdx.x;
    int block_idy = blockIdx.y;
    int offset;

    if(block_idx >= n_r_x || block_idy >= n_r_y) return;

    int region_idx = block_idx * stride + block_idy;

    region* target_region = r + region_idx;

    for(int step = 0; step < N_STEPS; ++ step){

        find_particles(target_region);

        if(threadIdx.x == 0) {
            for (int micro_step = 0; micro_step < target_region->n_shared_particles / NUM_THREADS + 1; ++micro_step) {
                apply_forces <<< 1, NUM_THREADS >>> (target_region, micro_step * NUM_THREADS);
                cudaThreadSynchronize();
            }
        }

        if(threadIdx.x == 0) {
            for (int micro_step = 0; micro_step < target_region->n_shared_particles / NUM_THREADS + 1; ++micro_step) {
                move_particles <<< 1, NUM_THREADS >>> (target_region, micro_step * NUM_THREADS);
                cudaThreadSynchronize();
            }
        }

        for(int b = 0; b < N_BLOCKS; ++b){
            if(b == 0){
                offset = 0;
            }else{
                offset += (target_region - 1)->n_shared_particles;
            }
            broadcast_locations(target_region, offset);
        }

    }

}

__device__ inline void get_idx(const double &x, const double &y,
                               const double &delta_x, const double &delta_y,
                               int &x_idx, int &y_idx){

    x_idx = (int) (x / delta_x);
    y_idx = (int) (y / delta_y);

}