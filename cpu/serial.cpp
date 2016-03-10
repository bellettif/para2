#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "frame.h"

#define cutoff  0.01
#define density 0.0005

//
//  benchmarking program x
//
int main( int argc, char **argv )
{

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

    int n = read_int( argc, argv, "-n", 1000 );

    int navg,nabsavg=0;
    double davg,dmin, absmin=1.0, absavg=0.0;

    int size = sqrt(density * n);

    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);

    FILE *fsave = savename ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname ? fopen(sumname, "a") : NULL;

    particle_t *particles = (particle_t *) malloc(n * sizeof(particle_t));
    set_size(n);
    init_particles(n, particles);

    double width = 0.0;
    double height = 0.0;
    for (int i = 0; i < n; ++i) {
        if (particles[i].x > width) {
            width = particles[i].x;
        }
        if (particles[i].y > height) {
            height = particles[i].y;
        }
    }

    std::cout << sqrt(n) / 10 << std::endl;

    Frame grid(sqrt(n) / 2, sqrt(n) / 2, particles, n);

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();

    for (int step = 0; step < NSTEPS; step++) {
        navg = 0;
        davg = 0.0;
        dmin = 1.0;

        //
        //  compute forces
        //
        for (int i = 0; i < n; i++) {
            particles[i].ax = particles[i].ay = 0;
            grid.apply_forces(particles[i], &dmin, &davg, &navg);
        }

        std::cout << "navg: " << navg << std::endl;
        std::cout << "davg: " << davg << std::endl;
        std::cout << "dmin: " << dmin << std::endl;

        //
        //  move particles
        //
        for (int i = 0; i < n; i++) {
            move(particles[i]);
        }

        grid.update_locations(particles, n);

        if (find_option(argc, argv, "-no") == -1) {
            //
            // Computing statistical data
            //
            if (navg) {
                absavg += davg / navg;
                nabsavg++;
            }
            if (dmin < absmin) absmin = dmin;

            //
            //  save if necessary
            //
            if (fsave && (step % SAVEFREQ) == 0)
                save(fsave, n, particles);
        }
    }
    simulation_time = read_timer() - simulation_time;

    printf("\nn = %d, simulation time = %g seconds", n, simulation_time);

    if (find_option(argc, argv, "-no") == -1) {
        if (nabsavg) absavg /= nabsavg;
        //
        //  -The minimum distance absmin between 2 particles during the run of the simulation
        //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
        //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
        //
        //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
        //
        printf(", absmin = %lf, absavg = %lf", absmin, absavg);
        if (absmin < 0.4)
            printf("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
        if (absavg < 0.8)
            printf("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");

    //
    // Printing summary data
    //
    if (fsum)
        fprintf(fsum, "%d %g\n", n, simulation_time);

    //
    // Clearing space
    //
    if (fsum)
        fclose(fsum);
    free(particles);
    if (fsave)
        fclose(fsave);

    return 0;
}
