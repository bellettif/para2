#pragma once

#include<cuda.h>

#include "physics.cuh"
#include "common.h"

typedef struct {

    particle_t *d_local_particles;
    particle_t *h_local_particles;
    int n_local_particles;

    particle_t *d_helper_particles;
    particle_t *h_helper_particles;
    int n_helper_particles;

} region;

void assign_particles(region *rs,
                      const int block_stride,
                      const int n_block_x,
                      const int n_block_y,
                      const double delta_x, const double delta_y,
                      particle_t *particles, const int n_particles);

void copy_region_to_device(region &r);

void copy_region_to_host(region &r);

void copy_regions_to_array(region *regions, const int n_regions, particle_t * particles, int n);



