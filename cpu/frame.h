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


#pragma once

#include "common.h"
#include <stdlib.h>
#include <vector>
#include <assert.h>
#include <iostream>

#define density 0.0005

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }
static inline double abs(double x) { return (x < .0 ? -x : x); }

class Frame
{

public:
    Frame(const int _n_x, const int _n_y,
          particle_t* particles, const int _n_particles):
            size(sqrt( density * _n_particles)),
            delta_x(sqrt( density * _n_particles) / ((double) _n_x)),
            delta_y(sqrt( density * _n_particles) / ((double) _n_y)),
            n_x(_n_x),
            n_y(_n_y),
            n_particles(_n_particles){

        part_grid = new particle_t***[n_x];
        next_part_grid = new particle_t***[n_x];

        size_grid = new int*[n_x];
        next_size_grid = new int*[n_x];

        int i, j;

        for(i = 0; i < n_x; ++i){
            part_grid[i] = new particle_t**[n_y];
            next_part_grid[i] = new particle_t**[n_y];

            size_grid[i] = new int[n_y];
            next_size_grid[i] = new int[n_y];

            for(j = 0; j < n_y; ++j){
                size_grid[i][j] = -1;
                next_size_grid[i][j] = -1;
            }

        }

        update_locations(particles, n_particles, false);

    }

    inline void update_locations(particle_t* particles, const int n_particles, bool next_frame = true) {

        particle_t ****target_grid;
        int **target_size_grid;
        if(next_frame) {
            target_grid = next_part_grid;
            target_size_grid = next_size_grid;
        }else{
            target_grid = part_grid;
            target_size_grid = size_grid;
        }

        int x_idx, y_idx;
        for (int i = 0; i < n_particles; ++i) {
            get_idx(particles[i].x, particles[i].y, x_idx, y_idx);
            int &size_x_y = target_size_grid[x_idx][y_idx];
            if (size_x_y == -1) {
                target_grid[x_idx][y_idx] = new particle_t *[n_particles];
                size_x_y = 0;
            }
            target_grid[x_idx][y_idx][size_x_y] = particles + i;
            ++size_x_y;
        }

        if(next_frame){
            for(int i = 0; i < n_x; ++i){
                for(int j = 0; j < n_y; ++j){
                    size_grid[i][j] = min(0, size_grid[i][j]);
                }
            }

            swap_part_grid = part_grid;
            swap_size_grid = size_grid;

            part_grid = target_grid;
            size_grid = target_size_grid;

            next_part_grid = swap_part_grid;
            next_size_grid = swap_size_grid;

        }

        /*
        int current_size;
        for(int i = 0; i < n_x; ++i){
            for(int j = 0; j < n_y; ++j){
                current_size = size_grid[i][j];
                if(current_size > 0){
                    for(int k = 0; k < current_size; ++k){
                        get_idx(part_grid[i][j][k]->x, part_grid[i][j][k]->y, x_idx, y_idx);
                        assert(i == x_idx);
                        assert(j == y_idx);
                    }
                }
            }
        }
        */

    }

    inline void apply_forces(particle_t& part,
                             double *dmin,
                             double *davg,
                             int *navg){

        int x_idx, y_idx, size_x_y;
        particle_t** ptr;
        get_idx(part.x, part.y, x_idx, y_idx);

        size_x_y = size_grid[x_idx][y_idx];
        ptr = part_grid[x_idx][y_idx];
        for(int i = 0; i < size_x_y; ++i){
            apply_force(part, **(ptr++), dmin, davg, navg);
        }

        // South west neighboring cell
        if(x_idx > 0 && y_idx > 0){
            size_x_y = size_grid[x_idx - 1][y_idx - 1];
            ptr = part_grid[x_idx - 1][y_idx - 1];
            for(int i = 0; i < size_x_y; ++i){
                apply_force(part, **(ptr++), dmin, davg, navg);
            }
        }

        // South neighboring cell
        if(y_idx > 0){
            size_x_y = size_grid[x_idx][y_idx - 1];
            ptr = part_grid[x_idx][y_idx - 1];
            for(int i = 0; i < size_x_y; ++i){
                apply_force(part, **(ptr++), dmin, davg, navg);
            }
        }

        // Shout east neighboring cell
        if(x_idx < n_x - 1 && y_idx > 0){
            size_x_y = size_grid[x_idx + 1][y_idx - 1];
            ptr = part_grid[x_idx + 1][y_idx - 1];
            for(int i = 0; i < size_x_y; ++i){
                apply_force(part, **(ptr++), dmin, davg, navg);
            }
        }

        // West neighboring cell
        if(x_idx > 0){
            size_x_y = size_grid[x_idx - 1][y_idx];
            ptr = part_grid[x_idx - 1][y_idx];
            for(int i = 0; i < size_x_y; ++i){
                apply_force(part, **(ptr++), dmin, davg, navg);
            }
        }

        // East neighboring cell
        if(x_idx < n_x - 1){
            size_x_y = size_grid[x_idx + 1][y_idx];
            ptr = part_grid[x_idx + 1][y_idx];
            for(int i = 0; i < size_x_y; ++i){
                apply_force(part, **(ptr++), dmin, davg, navg);
            }
        }

        // North west neighboring cell
        if(x_idx > 0 && y_idx < n_y - 1){
            size_x_y = size_grid[x_idx - 1][y_idx + 1];
            ptr = part_grid[x_idx - 1][y_idx + 1];
            for(int i = 0; i < size_x_y; ++i){
                apply_force(part, **(ptr++), dmin, davg, navg);
            }
        }

        // North neighboring cell
        if(y_idx < n_y - 1){
            size_x_y = size_grid[x_idx][y_idx + 1];
            ptr = part_grid[x_idx][y_idx + 1];
            for(int i = 0; i < size_x_y; ++i){
                apply_force(part, **(ptr++), dmin, davg, navg);
            }
        }

        // North east neighboring cell
        if(x_idx < n_x - 1 && y_idx < n_y - 1){
            size_x_y = size_grid[x_idx + 1][y_idx + 1];
            ptr = part_grid[x_idx + 1][y_idx + 1];
            for(int i = 0; i < size_x_y; ++i){
                apply_force(part, **(ptr++), dmin, davg, navg);
            }
        }

    }

    virtual void const print(){
        double loc_density;
        for(int i = 0; i < n_x; ++i){
            for(int j = 0; j < n_y; ++j){
                loc_density = ((double) size_grid[i][j]) / ((double) n_particles);
                if(loc_density < 0.0) {
                    std::cout << ".";
                    continue;
                }
                if(loc_density == 0.0) {
                    std::cout << "0";
                    continue;
                }
                if(loc_density <= 0.05) {
                    std::cout << "*";
                    continue;
                }
                if(loc_density <= 0.10) {
                    std::cout << "#";
                    continue;
                }
                if(loc_density <= 0.20) {
                    std::cout << "@";
                    continue;
                }
                std::cout << "!";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

protected:
    const double delta_x;
    const double delta_y;
    const double size;
    const int n_x;
    const int n_y;
    const int n_particles;

    particle_t**** part_grid;
    int** size_grid;

    particle_t**** next_part_grid;
    int** next_size_grid;

    particle_t**** swap_part_grid;
    int** swap_size_grid;

    /**
    * Look up the indices of that location.
    * IN: x, y. OUT: x_idx, y_idx
    */
    inline void get_idx(const double &x, const double &y, int &x_idx, int &y_idx){
        x_idx = (int) (x / delta_x);
        y_idx = (int) (y / delta_y);
    }


};