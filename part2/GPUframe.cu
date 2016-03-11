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

#define BUFF_SIZE 512

__global__ void simulate(region* r, const int stride,
                         const int n_r_x, int const n_r_y,
                         const double size){

    int block_idx = blockIdx.x;
    int block_idy = blockIdx.y;

    //if(block_idx >= n_r_x || block_idy >= n_r_y) return;

    int region_idx = block_idx * stride + block_idy;

    region* target_region = r + region_idx;

    int offset;
    int i;

    __shared__ particle_t helper_mem[BUFF_SIZE];

    int n_help;

    particle_t *core_mem = target_region->d_local_particles;

    for(int macro_step = 0; macro_step < target_region->n_helper_particles / BUFF_SIZE + 1; ++ macro_step) {

        if(BUFF_SIZE * macro_step > target_region->n_helper_particles) break;

        n_help = min(target_region->n_helper_particles - BUFF_SIZE * macro_step, BUFF_SIZE);

        for (int micro_step = 0; micro_step < n_help / NUM_THREADS + 1; ++micro_step) {
            offset = micro_step * NUM_THREADS;
            i = offset + threadIdx.x;
            if (i >= n_help) break;
            helper_mem[i] = *(target_region->d_helper_particles + BUFF_SIZE * macro_step + i);
        }

        __syncthreads();


        assert(target_region->n_local_particles != 0);

        for (int micro_step = 0; micro_step < (target_region->n_local_particles / NUM_THREADS) + 1; ++micro_step) {
            offset = micro_step * NUM_THREADS;
            i = offset + threadIdx.x;
            if (i >= target_region->n_local_particles) break;
            particle_t *part = core_mem + i;

            if(macro_step == 0) {
                part->ax = 0.0;
                part->ay = 0.0;
            }

            for (int j = 0; j < n_help; ++j) {
                double dx = helper_mem[j].x - part->x;
                double dy = helper_mem[j].y - part->y;
                double r2 = dx * dx + dy * dy;
                if (r2 > cutoff * cutoff)
                    continue;
                //r2 = fmax( r2, min_r*min_r );
                r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
                double r = sqrt(r2);

                //
                //  very simple short-range repulsive force
                //
                double coef = (1 - cutoff / r) / r2 / mass;
                part->ax += coef * dx;
                part->ay += coef * dy;
            }

        }

    }

    __syncthreads();

    for (int micro_step = 0; micro_step < (target_region->n_local_particles / NUM_THREADS) + 1; ++micro_step) {
        offset = micro_step * NUM_THREADS;
        i = offset + threadIdx.x;
        if (i >= target_region->n_local_particles) break;

        particle_t *p = core_mem + i;

        //
        //  slightly simplified Velocity Verlet integration
        //  conserves energy better than explicit Euler method
        //
        p->vx += p->ax * dt;
        p->vy += p->ay * dt;
        p->x += p->vx * dt;
        p->y += p->vy * dt;

        //
        //  bounce from walls
        //
        while (p->x < 0 || p->x > size) {
            p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
            p->vx = -(p->vx);
        }
        while (p->y < 0 || p->y > size) {
            p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
            p->vy = -(p->vy);
        }

    }

}