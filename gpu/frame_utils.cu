#include "frame_utils.cuh"
#include "common.h"

void assign_particles(region *rs,
                      const int block_stride,
                      const int n_block_x,
                      const int n_block_y,
                      const double delta_x, const double delta_y,
                      particle_t *particles, const int n_particles){

    for(int i = 0; i < n_block_x; ++i){
        for(int j = 0; j < n_block_y; ++j){
            rs[i * block_stride + j].n_local_particles = 0;
            rs[i * block_stride + j].n_helper_particles = 0;
        }
    }

    int idx, idy;
    for(int i = 0; i < n_particles; ++i){
        const particle_t &part = particles[i];

        idx = (int) (part.x / delta_x);
        idy = (int) (part.y / delta_y);

        for(int x_offset = -1; x_offset <= 1; ++ x_offset){
            if(idx + x_offset >= 0 && idx + x_offset < n_block_x) {
                for (int y_offset = -1; y_offset <= 1; ++y_offset) {
                    if (idy + y_offset >= 0 && idy + y_offset < n_block_y){
                        region & target_region = rs[(idx + x_offset) * block_stride + (idy + y_offset)];
                        if(idx == idy){
                            target_region.h_local_particles[target_region.n_local_particles++] = part;
                        }else{
                            target_region.h_helper_particles[target_region.n_helper_particles++] = part;
                        }
                    }
                }
            }
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



