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


#include "GPUframe.cuh"

__global__ void apply_forces(region *r, const int offset){

    int i = threadIdx.x;

    if(i >= r->n_shared_particles - offset) return;

    particle_t& part = *(r->shared_particles + offset + i);
    part.ax = 0;
    part.ay = 0;

    for(int j = 0; j < r->n_shared_particles; ++j){
        apply_force_gpu(part, r->shared_particles[j]);
    }

}

__global__ void find_particles(region *r){

    int n_shared_particles = 0;

    for(int i = 0; i < r->n_global_particles; ++i) {
        particle_t part = r->global_particles[i];
        if (part.x >= r->x_min && part.x < r->x_max && part.y >= r->y_min && part.y < r->y_max) {
            r->shared_particles[n_shared_particles] = part;
            r->shared_particle_idx[n_shared_particles] = i;
            ++ n_shared_particles;
        }
    }

    r->n_shared_particles = n_shared_particles;

}

__global__ void broadcast_locations(region *r, const int offset){

    int i = threadIdx.x;

    if(i >= r->n_global_particles - offset) return;

    r->global_particles[r->shared_particle_idx[i + offset]] = r->shared_particles[i + offset];

}

__global__ void simulate(region* r, const int stride,
                         const int n_r_x, int const n_r_y,
                         const double size){

    int block_idx = blockIdx.x;
    int block_idy = blockIdx.y;

    if(block_idx >= n_r_x || block_idy >= n_r_y) return;

    int region_idx = block_idx * stride + block_idy;

    region* target_region = r + region_idx;

    if(threadIdx.x == 0) find_particles <<<1, 1>>>(target_region);

    if(threadIdx.x == 0) {
        for (int micro_step = 0; micro_step < target_region->n_shared_particles / NUM_THREADS + 1; ++micro_step) {
            apply_forces <<< 1, NUM_THREADS >>> (target_region, micro_step * NUM_THREADS);
            __syncthreads();
        }
    }

    if(threadIdx.x == 0) {
        for (int micro_step = 0; micro_step < target_region->n_shared_particles / NUM_THREADS + 1; ++micro_step) {
            move_gpu <<< 1, NUM_THREADS >>> (
                target_region->shared_particles + micro_step * NUM_THREADS,
                target_region->n_shared_particles + micro_step * NUM_THREADS,
                size);
            __syncthreads();
        }
    }

    if(threadIdx.x == 0) {
        for (int micro_step = 0; micro_step < target_region->n_shared_particles / NUM_THREADS + 1; ++micro_step) {
            broadcast_locations <<< 1, NUM_THREADS >>> (target_region, micro_step * NUM_THREADS);;
            __syncthreads();
        }
    }

}