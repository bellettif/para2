#pragma once

#include<cuda.h>

#include "physics.cuh"
#include "common.h"

typedef struct {
    double x_min;
    double x_max;
    double y_min;
    double y_max;

    double helper_x_min;
    double helper_x_max;
    double helper_y_min;
    double helper_y_max;

    particle_t *d_local_particles;
    particle_t *h_local_particles;
    int n_local_particles;

    particle_t *d_helper_particles;
    particle_t *h_helper_particles;
    int n_helper_particles;

} region;

void assign_particles(region &r, particle_t *particles, const int n_particles);

void copy_region_to_device(region &r);

void copy_region_to_host(region &r);

void copy_regions_to_array(region *regions, const int n_regions, particle_t * particles, int n);



