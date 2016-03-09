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
#include "frame_utils.cuh"

#define BUFF_SIZE 256

__global__ void simulate(region* r, const int stride,
                         const int n_r_x, int const n_r_y,
                         const double size){

    int block_idx = blockIdx.x;
    int block_idy = blockIdx.y;

    if(block_idx >= n_r_x || block_idy >= n_r_y) return;

    int region_idx = block_idx * stride + block_idy;

    region* target_region = r + region_idx;

    __shared__ particle_t core_mem[BUFF_SIZE];
    __shared__ particle_t helper_mem[BUFF_SIZE];

    int offset;
    int i;
    for (int micro_step = 0; micro_step < target_region->n_local_particles / NUM_THREADS + 1; ++micro_step) {
        offset = micro_step * NUM_THREADS;
        i = offset + threadIdx.x;
        if(i >= target_region->n_local_particles) break;
        core_mem[i] = target_region->d_local_particles[i];
    }

    for (int micro_step = 0; micro_step < target_region->n_helper_particles / NUM_THREADS + 1; ++micro_step) {
        offset = micro_step * NUM_THREADS;
        i = offset + threadIdx.x;
        if(i >= target_region->n_helper_particles) break;
        helper_mem[i] = target_region->d_helper_particles[i];
    }

    for (int micro_step = 0; micro_step < target_region->n_local_particles / NUM_THREADS + 1; ++micro_step) {
        offset = micro_step * NUM_THREADS;
        i = offset + threadIdx.x;
        if(i >= target_region->n_local_particles) break;
        particle_t *part = core_mem + i;

        part->ax = 0.0;
        part->ay = 0.0;

        for(int j = 0; j < r->n_local_particles; ++j){
            double dx = core_mem[j].x - part->x;
            double dy = core_mem[j].y - part->y;
            double r2 = dx * dx + dy * dy;
            if( r2 > cutoff*cutoff )
                return;
            //r2 = fmax( r2, min_r*min_r );
            r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
            double r = sqrt( r2 );

            //
            //  very simple short-range repulsive force
            //
            double coef = ( 1 - cutoff / r ) / r2 / mass;
            part->ax += coef * dx;
            part->ay += coef * dy;
        }

        for(int j = 0; j < r->n_helper_particles; ++j){
            double dx = helper_mem[j].x - part->x;
            double dy = helper_mem[j].y - part->y;
            double r2 = dx * dx + dy * dy;
            if( r2 > cutoff*cutoff )
                return;
            //r2 = fmax( r2, min_r*min_r );
            r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
            double r = sqrt( r2 );

            //
            //  very simple short-range repulsive force
            //
            double coef = ( 1 - cutoff / r ) / r2 / mass;
            part->ax += coef * dx;
            part->ay += coef * dy;
        }

    }

    for (int micro_step = 0; micro_step < target_region->n_local_particles / NUM_THREADS + 1; ++micro_step) {
        offset = micro_step * NUM_THREADS;
        i = offset + threadIdx.x;
        if(i >= target_region->n_local_particles) break;
        target_region->d_local_particles[i] = core_mem[i];
    }

}