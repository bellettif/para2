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
    
    int n = read_int( argc, argv, "-n", 100000);

    char *savename = read_string( argc, argv, "-o", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );

    init_particles( n, particles );

    printf("\n");

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;
    
    //
    //  simulate a number of time steps
    //
    double size = sqrt(density * n );
    double simulation_time = read_timer( );

    int n_block_x = max((int) (size / (50 * cutoff)), 1);
    int n_block_y = max((int) (size / (50 * cutoff)), 1);

    const double delta = size / ((double) n_block_x);

    int block_stride = n_block_y;

    printf("\nDelta = %f, size = %f", delta, size);
    printf("\nN block x = %d, n block y = %d, stride = %d", n_block_x, n_block_y, block_stride);

    region all_regions[n_block_x * n_block_y];

    for(int i = 0; i < n_block_x; ++i){
        for(int j = 0; j < n_block_y; ++j){
            region &target = all_regions[i * block_stride + j];

            target.x_min = delta * i - cutoff;
            target.y_min = delta * j - cutoff;
            target.x_max = delta * (i + 1) + cutoff;
            target.y_max = delta * (j + 1) + cutoff;

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

    double dmin = 1.0;
    double davg;
    int navg;

    double absavg = 0.0;
    double absdmin = 1.0;

    int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
    for( int step = 0; step < NSTEPS; step++ )
    {
        davg = 0.0;
        navg = 0;
        dmin = 1.0;

        assign_particles(all_regions, n_block_y, n_block_x, n_block_y, delta, delta, particles, n);

        for(int i = 0; i < n_block_x * n_block_y; ++i){
            copy_region_to_device(all_regions[i]);
        }

        cudaMemcpy(dev_all_regions, all_regions, n_block_x * n_block_y * sizeof(region), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        simulate <<<numBlocks, NUM_THREADS>>> (dev_all_regions, block_stride, n_block_x, n_block_y, size);
        cudaDeviceSynchronize();

        for(int i = 0; i < n_block_x * n_block_y; ++i){
            copy_region_to_host(all_regions[i]);
        }
        cudaDeviceSynchronize();
        copy_regions_to_array(all_regions, n_block_x * n_block_y, particles, n);

        /*
        for(int i = 0; i < n; ++i){
            const particle_t &p1 = particles[i];
            for(int j = 0; j < n; ++j){
                const particle_t & p2 = particles[j];
                double dx = p1.x - p2.x;
                double dy = p1.y - p2.y;
                double r2 = dx * dx + dy * dy;
                if(r2 == 0.0) continue;
                if( r2 > cutoff*cutoff )
                    continue;
                if(r2 / (cutoff * cutoff) < dmin * dmin){
                    dmin = sqrt(r2) / cutoff;
                    davg += sqrt(r2) / cutoff;
                    ++ navg;
                }
            }
        }
        absavg += davg / max(navg, 1);
        if(dmin < absdmin) absdmin = dmin;
        */

        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
            //cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            //save( fsave, n, particles);
	}
    }
    cudaDeviceSynchronize();
    simulation_time = read_timer( ) - simulation_time;

    printf("\n%f %f\n", absdmin, absavg / NSTEPS);
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
