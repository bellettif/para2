#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "common.h"

#include "MPIVectframe.h"

#define cutoff  0.01

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg; 
 
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 500 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    std::cout << "Proc " << rank << std::endl;
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );

    particle_t* local_partition;
    int local_n_particles = 0;

    set_size( n );

    double size = sqrt(density * n);

    int block_stride = (int) sqrt(n_proc);
    int n_block_y = max(block_stride, 1);
    int n_block_x = max(n_proc / block_stride, 1);

    assert(n_proc == n_block_x * n_block_y);

    double block_delta_x = size / ((double) n_block_x);
    double block_delta_y = size / ((double) n_block_y);

    int n_x = max(block_delta_x / cutoff, 1.0);
    int n_y = max(block_delta_y / cutoff, 1.0);

    //n_x = 1;
    //n_y = 1;

    std::cout << "Size " << size << std::endl;
    std::cout << "Block stride = " << block_stride << std::endl;
    std::cout << "N block x = " << n_block_x << std::endl;
    std::cout << "N block y = " << n_block_y << std::endl;
    std::cout << "Block delta x = " << block_delta_x << std::endl;
    std::cout << "Block delta y = " << block_delta_y << std::endl;

    if( rank == 0 ) {

        init_particles(n, particles);

        particle_t** partitions = new particle_t*[n_proc];
        int* n_particles = new int[n_proc];
        for(int p = 0; p < n_proc; ++p){
            partitions[p] = new particle_t[n];
            n_particles[p] = 0;
        }

        int x_idx, y_idx, proc_idx;

        for (int i = 0; i < n; ++i) {
            const particle_t &part = particles[i];

            x_idx = (int) (part.x / block_delta_x);
            y_idx = (int) (part.y / block_delta_y);
            proc_idx = x_idx * block_stride + y_idx;

            partitions[proc_idx][n_particles[proc_idx]++] = part;
        }

        local_n_particles = n_particles[0];
        local_partition = new particle_t[local_n_particles];
        for(int i = 0; i < n_particles[0]; ++i){
            local_partition[i] = partitions[0][i];
        }

        MPI_Request reqs[n_proc - 1];
        MPI_Status  status[n_proc - 1];
        for(int p = 1; p < n_proc; ++p){
            MPI_Isend(&n_particles[p], 1, MPI_INT, p, 2 * p, MPI_COMM_WORLD, reqs  + p - 1);
        }
        //std::cout << "O Sending initial size data from " << rank << std::endl;
        MPI_Waitall(n_proc - 1, reqs, status);
        //std::cout << "X Sent initial size data from " << rank << std::endl;

        for(int p = 1; p < n_proc; ++p){
            MPI_Isend(partitions[p], n_particles[p], PARTICLE, p, 2 * p + 1, MPI_COMM_WORLD, reqs + p - 1);
        }

        //std::cout << "O Sending initial particle data from " << rank << std::endl;
        MPI_Waitall(n_proc - 1, reqs, status);
        //std::cout << "X Sent initial particle data from " << rank << std::endl;

    }else{

        MPI_Request req;

        MPI_Irecv(&local_n_particles, 1, MPI_INT, 0, 2 * rank, MPI_COMM_WORLD, &req);
        //std::cout << "O Receiving initial size data on " << rank << std::endl;
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        //std::cout << "X Received initial size data on " << rank << std::endl;

        local_partition = new particle_t[local_n_particles];

        MPI_Irecv(local_partition, local_n_particles, PARTICLE, 0, 2 * rank + 1, MPI_COMM_WORLD, &req);
        //std::cout << "O Receiving initial particle data on " << rank << std::endl;
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        //std::cout << "X Received initial particle data on " << rank << std::endl;

    }

    int block_x = rank / block_stride;
    int block_y = rank % block_stride;
    //std::cout << "Starting init on " << rank << std::endl;

    assert(block_delta_x != 0.0);
    assert(block_delta_y != 0.0);

    assert(rank == block_x * block_stride + block_y);

    MPIVectFrame frame(block_stride,
                   block_x,
                   block_y,
                   n_block_x,
                   n_block_y,
                   size,
                   block_delta_x,
                   block_delta_y,
                   n_x,
                   n_y,
                   local_partition,
                   local_n_particles,
                   n);
    //std::cout << "Init done on " << rank << std::endl;
    //std::cout << "Block x = " << block_x << std::endl;
    //std::cout << "Block y = " << block_y << std::endl;
    //std::cout << std::endl;

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {

        //std::cout << "------------ STARTING STEP " << step << " ON " << rank << std::endl;

        // 
        //  collect all global data locally (not good idea to do)
        //
        //MPI_Allgatherv( local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD );
        
        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( find_option( argc, argv, "-no" ) == -1 )
          if( fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
        
        //
        //  compute all forces
        //
        //std::cout << step << ": Applying forces on " << rank << std::endl;
        frame.apply_forces(step);
        //std::cout << step << ": Applied forces on " << rank << std::endl;

        //
        //  move particles
        //
        //std::cout << step << ": Updating locations on " << rank << std::endl;
        frame.update_locations(step);
        //std::cout << step << ": Updated locations on " << rank << std::endl;

        if( find_option( argc, argv, "-no" ) == -1 )
        {

            rnavg = 0;
            rdmin = 1.0;
            rdavg = 0.0;

            //std::cout << step << ": davg on " << rank << ":" << frame.davg << std::endl;
            //std::cout << step << ": navg on " << rank << ":" << frame.navg << std::endl;
            //std::cout << step << ": dmin on " << rank << ":" << frame.dmin << std::endl;

            MPI_Reduce(&frame.davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&frame.navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&frame.dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

            //assert(rdmin > 0.4);

            if (rank == 0){

                std::cout << step << ": rdavg on " << rank << ":" << rdavg << std::endl;
                std::cout << step << ": rnavg on " << rank << ":" << rnavg << std::endl;
                std::cout << step << ": rdmin on " << rank << ":" << rdmin << std::endl;

                //
                // Computing statistical data
                //
                if (rnavg) {
                    absavg +=  rdavg/rnavg;
                    nabsavg++;
                }
                //assert(rdmin != 0.0);
                if (rdmin < absmin) absmin = rdmin;
                //assert(absmin >= 0.1);
                //std::cout << "Done computing statistical data " << rank << std::endl;
            }

        }

        //std::cout << "------------ DONE WITH STEP " << step << " ON " << rank << std::endl;

    }
    simulation_time = read_timer( ) - simulation_time;
  
    if (rank == 0) {  
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

      if( find_option( argc, argv, "-no" ) == -1 )
      {
        if (nabsavg) absavg /= nabsavg;
      // 
      //  -The minimum distance absmin between 2 particles during the run of the simulation
      //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
      //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
      //
      //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
      //
      printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
      if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
      if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
      }
      printf("\n");     
        
      //  
      // Printing summary data
      //  
      if( fsum)
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    free( particles );
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}
