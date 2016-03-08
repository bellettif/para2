#include "physics.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <math.h>
#include <cuda.h>

#include "common.h"
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

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );

    init_particles( n, particles );

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;
    
    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    double size = sqrt(density * n );

    int n_block_x = size / cutoff;
    int n_block_y = size / cutoff;
    int block_stride = n_block_y;

    region all_regions[n_block_x * n_block_y];

    for(int i = 0; i < n_block_x; ++i){
        for(int j = 0; j < n_block_y; ++j){
            region &target = all_regions[i * block_stride + j];
            target.x_min = cutoff * i;
            target.x_max = cutoff * (i + 1);
            target.y_min = cutoff * j;
            target.y_max = cutoff * (j + 1);
            target.global_particles = d_particles;
            target.n_global_particles = n;
        }
    }

    region *dev_all_regions;
    cudaMalloc((void **) &dev_all_regions, n_block_x * n_block_y * sizeof(region));
    cudaThreadSynchronize();

    cudaMemcpy(dev_all_regions, all_regions, n_block_x * n_block_y * sizeof(region), cudaMemcpyHostToDevice);
    cudaThreadSynchronize();

    dim3 numBlocks(n_block_x, n_block_y);

    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  compute forces
        //

	//int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
	//compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, n);
        
        //
        //  move particles
        //
	//move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);

        simulate <<<numBlocks, NUM_THREADS>>> (dev_all_regions, block_stride, n_block_x, n_block_y, size);

        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
	}
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
