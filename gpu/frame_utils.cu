#include "frame_utils.cuh"
#include "common.h"

void assign_particles(region &r, particle_t *particles, const int n_particles){

    r.n_local_particles = 0;
    r.n_helper_particles = 0;

    for(int i = 0; i < n_particles; ++i){
        const particle_t &part = particles[i];
        if((part.x >= r.x_min) && (part.x < r.x_max) && (part.y >= r.y_min) && (part.y < r.y_max)){
            r.h_local_particles[r.n_local_particles ++] = part;
        }
        if((part.x >= r.helper_x_min) && (part.x <= r.helper_x_max) && (part.y >= r.helper_y_min) && (part.y <= r.helper_y_max)){
            r.h_helper_particles[r.n_helper_particles ++] = part;
        }
    }

}

void copy_region_to_device(region &r){
    cudaMemcpy(r.d_local_particles, r.h_local_particles, r.n_local_particles * sizeof(particle_t), cudaMemcpyHostToDevice);
    cudaMemcpy(r.d_helper_particles, r.h_helper_particles, r.n_helper_particles * sizeof(particle_t), cudaMemcpyHostToDevice);
}

void copy_region_to_host(region &r){
    cudaMemcpy(r.h_local_particles, r.d_local_particles, r.n_local_particles * sizeof(particle_t), cudaMemcpyDeviceToHost);
}

void copy_regions_to_array(region *regions, const int n_regions, particle_t * particles, int n){
    int old_n = n;

    n = 0;
    for(int i = 0; i < n_regions; ++i) {
        const region &r = regions[i];
        for (int j = 0; j < r.n_local_particles; ++j) {
            particles[n] = r.h_local_particles[j];
            ++n;
        }
    }

    assert(old_n == n);
}



