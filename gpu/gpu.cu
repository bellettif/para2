#include "physics.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <math.h>
#include <cuda.h>

#include "common.h"
#include "frame_utils.cuh"
#include "GPUframe.cuh"

int main( int argc, char **argv )
{    
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize(); 

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    particle_t *h_check = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    particle_t * d_check;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));
    cudaMalloc((void **) &d_check, n * sizeof(particle_t));

    set_size( n );

    init_particles( n, particles );

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_check, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;
    
    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    double size = sqrt(density * n );

    const double delta = cutoff * 20;

    int n_block_x = (int) ((size + delta) / delta);
    int n_block_y = (int) ((size + delta) / delta);
    int block_stride = n_block_y;

    printf("\nDelta = %f, size = %f", delta, size);
    printf("\nN block x = %d, n block y = %d, stride = %d", n_block_x, n_block_y, block_stride);

    region all_regions[n_block_x * n_block_y];

    for(int i = 0; i < n_block_x; ++i){
        for(int j = 0; j < n_block_y; ++j){
            region &target = all_regions[i * block_stride + j];
            target.x_min = delta * i;
            target.x_max = delta * (i + 1);
            target.y_min = delta * j;
            target.y_max = delta * (j + 1);

            target.helper_x_min = max(delta * (i - 1), 0.0);
            target.helper_x_max = min(delta * (i + 2), size);
            target.helper_y_min = max(delta * (j - 1), 0.0);
            target.helper_y_max = min(delta * (j + 2), size);

            target.n_local_particles = 0;
            target.h_local_particles = (particle_t*) malloc( n * sizeof(particle_t) );
            cudaMalloc((void **) &target.d_local_particles, n * sizeof(particle_t));

            target.n_helper_particles = 0;
            target.h_helper_particles = (particle_t*) malloc( n * sizeof(particle_t) );
            cudaMalloc((void **) &target.d_helper_particles, n * sizeof(particle_t));

        }
    }

    region *dev_all_regions;
    cudaMalloc((void **) &dev_all_regions, n_block_x * n_block_y * sizeof(region));
    cudaThreadSynchronize();

    dim3 numBlocks(n_block_x, n_block_y);

    int blks = (n + NUM_THREADS - 1) / NUM_THREADS;

    /*
    for( int step = 0; step < NSTEPS; step++ )
    {

        compute_forces_gpu <<< blks, NUM_THREADS >>> (d_check, n);
        move_gpu <<< blks, NUM_THREADS >>> (d_check, n, size);


    }
    cudaThreadSynchronize();
    cudaMemcpy(h_check, d_check, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
    */

    for( int step = 0; step < NSTEPS; step++ )
    {

        for(int i = 0; i < n_block_x * n_block_y; ++i){
            assign_particles(all_regions[i], particles, n);
            copy_region_to_device(all_regions[i]);
        }

        cudaMemcpy(dev_all_regions, all_regions, n_block_x * n_block_y * sizeof(region), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        simulate <<<numBlocks, NUM_THREADS>>> (dev_all_regions, block_stride, n_block_x, n_block_y, size);
        cudaDeviceSynchronize();

        //
        //  move particles
        //
        move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
        cudaDeviceSynchronize();

        for(int i = 0; i < n_block_x * n_block_y; ++i){
            copy_region_to_host(all_regions[i]);
        }

        copy_regions_to_array(all_regions, n_block_x * n_block_y, particles, n);

        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
	}
    }
    cudaDeviceSynchronize();
    simulation_time = read_timer( ) - simulation_time;

    cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /*
    for(int i = 0; i < n; ++i){
        printf("'%f %f %f %f\n", h_check[i].x, h_check[i].y, particles[i].x, particles[i].y);
        assert(h_check[i].x == particles[i].x);
        assert(h_check[i].y == particles[i].y);
    }
    */
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
