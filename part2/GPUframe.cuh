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

#include <cuda.h>

#include "frame_utils.cuh"

#define density 0.0005

__global__ void simulate(region* r, const int stride,
                         const int n_r_x, int const n_r_y,
                         const double size);