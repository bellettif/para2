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

#include "MPIComm.h"

//#define COMM_DEBUG
//#define SIM_DEBUG
//#define GEN_DEBUG
#define CHECK_ASSERT

#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005


class MPIVectFrame{

public:
    MPIVectFrame(const int _block_stride,
             const int _block_x, const int _block_y,
             const int _n_block_x, const int _n_block_y,
             particle_t *particles, const int _n_particles,
             const int tot_n_particles) :
            size(sqrt(density * tot_n_particles)),
            delta_x(cutoff),
            delta_y(cutoff),
            n_x(sqrt(density * tot_n_particles) / cutoff * _n_block_x),
            n_y(sqrt(density * tot_n_particles) / cutoff * _n_block_y),
            max_n_particles(tot_n_particles),
            x_min(_block_x * sqrt(density * tot_n_particles) / ((double) _n_block_x)),
            x_max((_block_x + 1) * sqrt(density * tot_n_particles) / ((double) _n_block_x)),
            y_min(_block_y * sqrt(density * tot_n_particles) / ((double) _n_block_y)),
            y_max((_block_y + 1) * sqrt(density * tot_n_particles) / ((double) _n_block_y)),
            rank(_block_x * _block_stride + _block_y),
            n_procs(_n_block_x * _n_block_y),
            block_stride(_block_stride),
            block_x(_block_x),
            block_y(_block_y),
            n_block_x(_n_block_x),
            n_block_y(_n_block_y),
            x_offset(_block_x * sqrt(density * tot_n_particles) / ((double) _n_block_x)),
            y_offset(_block_y * sqrt(density * tot_n_particles) / ((double) _n_block_y)),
            mpicom(rank)
    {

        navg = 0;
        dmin = 1.0;
        davg = 0.0;

#ifdef GEN_DEBUG
        std::cout << "-1: particles in frame " << _n_particles << " " << rank << std::endl;
        std::cout << "-1: offset " << x_offset << " " << y_offset << " over " << size << " on " << rank << std::endl;
#endif

        for(int i = 0; i < _n_particles; ++i){
            mem.push_back(particles[i]);
        }

#ifdef GEN_DEBUG
        std::cout << "-1: memsize in frame " << mem.size() << " " << rank << std::endl;
#endif

        msg_idx = 0;

        part_grid = new std::vector<particle_t*>*[n_x];
        next_part_grid = new std::vector<particle_t*>*[n_x];

        int i, j;

        for(i = 0; i < n_x; ++i){
            part_grid[i] = new std::vector<particle_t*>[n_y];
            next_part_grid[i] = new std::vector<particle_t*>[n_y];
        }

        part_send_buffers = new std::vector<particle_t>[n_procs];
        part_recv_buffers = new std::vector<particle_t>[n_procs];

        init_locations();

#ifdef GEN_DEBUG
        std::cout << "-0.5: memsize in frame " << mem.size() << " " << rank << std::endl;
#endif

    }

    /*
     * ------------------------------------------------
     *
     *              COMMUNICATION METHODS
     *
     * ------------------------------------------------
     *
     *
     * Send/receive information about particles in NW corner
     */

private:

    /**
     * Communicate particle locations so that
     * forces can be computed accurately for particles
     * near the border.
     */
    void comm_all_locations(int STEP){

#ifdef COMM_DEBUG
        std::cout << STEP << " :Communicating all locations on " << rank << std::endl;
#endif

        std::string comment = "locations";

        /*
         * Send to NW, receive from SE
         */
        mpicom.exchange_buffers(
                (block_x - 1) * block_stride + block_y + 1,
                (block_x + 1) * block_stride + block_y - 1,
                block_x > 0 && block_y < n_block_y - 1,
                block_x < n_block_x - 1 && block_y > 0,
                msg_idx,
                NWs_buffer,
                SEr_buffer,
                comment
        );

        /*
         * Send to N, receive from S
         */
        mpicom.exchange_buffers(
                block_x * block_stride + block_y + 1,
                block_x * block_stride + block_y - 1,
                block_y < n_block_y - 1,
                block_y > 0,
                msg_idx,
                Ns_buffer,
                Sr_buffer,
                comment
        );

        /*
         * Send to NE, receive from SW
         */
        mpicom.exchange_buffers(
                (block_x + 1) * block_stride + block_y + 1,
                (block_x - 1) * block_stride + block_y - 1,
                block_x < n_block_x - 1 && block_y < n_block_y - 1,
                block_x > 0 && block_y > 0,
                msg_idx,
                NEs_buffer,
                SWr_buffer,
                comment
        );

        /*
         * Send to W, receive from E
         */
        mpicom.exchange_buffers(
                (block_x - 1) * block_stride + block_y,
                (block_x + 1) * block_stride + block_y,
                block_x > 0,
                block_x < n_block_x - 1,
                msg_idx,
                Ws_buffer,
                Er_buffer,
                comment
        );

        /*
         * Send to E, receive from W
         */
        mpicom.exchange_buffers(
                (block_x + 1) * block_stride + block_y,
                (block_x - 1) * block_stride + block_y,
                block_x < n_block_x - 1,
                block_x > 0,
                msg_idx,
                Es_buffer,
                Wr_buffer,
                comment
        );

        /*
         * Send to SW, receive from NE
         */
        mpicom.exchange_buffers(
                (block_x - 1) * block_stride + block_y - 1,
                (block_x + 1) * block_stride + block_y + 1,
                block_x > 0 && block_y > 0,
                block_x < n_block_x - 1 && block_y < n_block_y - 1,
                msg_idx,
                SWs_buffer,
                NEr_buffer,
                comment
        );

        /*
         * Send to S, receive from N
         */
        mpicom.exchange_buffers(
                block_x * block_stride + block_y - 1,
                block_x * block_stride + block_y + 1,
                block_y > 0,
                block_y < n_block_y - 1,
                msg_idx,
                Ss_buffer,
                Nr_buffer,
                comment
        );

        /*
         * Send to SE, receive from NW
         */
        mpicom.exchange_buffers(
                (block_x + 1) * block_stride + block_y - 1,
                (block_x - 1) * block_stride + block_y + 1,
                block_x < n_block_x - 1 && block_y > 0,
                block_x > 0 && block_y < n_block_y - 1,
                msg_idx,
                SEs_buffer,
                NWr_buffer,
                comment
        );

#ifdef COMM_DEBUG
        std::cout << STEP << " :Communicated all locations on " << rank << std::endl;
#endif

    }


    /**
     * Communicate particles so that they are handled
     * by the processor managing the region they belong to.
     */
    void comm_all_particles(int STEP){

#ifdef COMM_DEBUG
        std::cout << STEP << " :Communicating all particles on " << rank << std::endl;
#endif

        std::string comment = "particles";

        int n_reqs = 2 * (n_procs - 1);
        MPI_Request reqs[n_reqs];
        MPI_Status status[n_reqs];

        int recv_buffer_sizes[n_procs];

        for(int p = 0; p < n_procs; ++p){
            if(p == rank) continue;
            mpicom.send_buffer_size(p, msg_idx, part_send_buffers[p], reqs[p - (p > rank)]);
        }

        for(int p = 0; p < n_procs; ++ p){
            if(p == rank) continue;
            mpicom.recv_buffer_size(p, msg_idx, recv_buffer_sizes[p], reqs[n_procs - 1 + p - (p > rank)]);
        }

        MPI_Waitall(n_reqs, reqs, status);
        ++msg_idx;

#ifdef COMM_DEBUG
        std::cout << STEP << ": Buff sizes on " << rank << ": ";
        for(int i = 0; i < n_procs; ++i){
            std::cout << " from " << i << "->" << recv_buffer_sizes[i] << " ";
        }std::cout << std::endl;

        std::cout << STEP << ": Send buffers on " << rank << ": ";
        for(int i = 0; i < n_procs; ++i){
            std::cout << "for " << i << " -> ";
            for(int j = 0; j < part_send_buffers[i].size(); ++j){
                std::cout << part_send_buffers[i][j].x << " " << part_send_buffers[i][j].y << " ";
            }
        }std::cout << std::endl;
#endif

        for(int p = 0; p < n_procs; ++p){
            if(p == rank) continue;
            mpicom.send_buffer(p, msg_idx, part_send_buffers[p], reqs[p - (p > rank)]);
        }

        for(int p = 0; p < n_procs; ++ p){
            if(p == rank) continue;
            part_recv_buffers[p].resize(recv_buffer_sizes[p]);
            mpicom.recv_buffer(p, msg_idx, part_recv_buffers[p], reqs[n_procs - 1 + p - (p > rank)]);
        }

        MPI_Waitall(n_reqs, reqs, status);
        ++msg_idx;

#ifdef COMM_DEBUG
        std::cout << STEP << " :Recv buffers on " << rank << ": ";
        for(int i = 0; i < n_procs; ++i){
            std::cout << "from " << i << " -> ";
            for(int j = 0; j < part_recv_buffers[i].size(); ++j){
                std::cout << part_recv_buffers[i][j].x << " " << part_recv_buffers[i][j].y << " ";
            }
        }std::cout << std::endl;
        std::cout << STEP << " :Communicated all particles on " << rank << std::endl;
#endif

    }


    /**
     * ----------------------------------------------------------
     *
     *                  SIMULATION METHODS
     *
     * ----------------------------------------------------------
     */

public:

    int navg;
    double dmin;
    double davg;
    int msg_idx;

    void apply_forces_from_buffers(int step, particle_t &part){

        // South West buffer
        for (int i = 0; i < SWr_buffer.size(); ++i) {
            apply_force(part, SWr_buffer.at(i), &dmin, &davg, &navg);
        }

        // South buffer
        for (int i = 0; i < Sr_buffer.size(); ++i) {
            apply_force(part, Sr_buffer.at(i), &dmin, &davg, &navg);
        }

        // South East buffer
        for (int i = 0; i < SEr_buffer.size(); ++i) {
            apply_force(part, SEr_buffer.at(i), &dmin, &davg, &navg);
        }

        // West buffer
        for (int i = 0; i < Wr_buffer.size(); ++i) {
            apply_force(part, Wr_buffer.at(i), &dmin, &davg, &navg);
        }

        // East buffer
        for (int i = 0; i < Er_buffer.size(); ++i) {
            apply_force(part, Er_buffer.at(i), &dmin, &davg, &navg);
        }

        // North West buffer
        for (int i = 0; i < NWr_buffer.size(); ++i) {
            apply_force(part, NWr_buffer.at(i), &dmin, &davg, &navg);
        }

        // North buffer
        for (int i = 0; i < Nr_buffer.size(); ++i) {
            apply_force(part, Nr_buffer.at(i), &dmin, &davg, &navg);
        }

        // North East buffer
        for (int i = 0; i < NEr_buffer.size(); ++i) {
            apply_force(part, NEr_buffer.at(i), &dmin, &davg, &navg);
        }

    }

    void apply_forces(int step){

        navg = 0;
        dmin = 1.0;
        davg = 0.0;

#ifdef SIM_DEBUG
        std::cout << step << ": mem_size for apply forces " << mem.size() << " on " << rank << std::endl;
#endif


        for(int p_idx = 0; p_idx < mem.size(); ++p_idx) {

            particle_t &part = mem.at(p_idx);
            part.ax = 0;
            part.ay = 0;

            int x_idx, y_idx, size_x_y;
            particle_t *ptr;

#ifdef CHECK_ASSERT
            assert(part.x >= x_min);
            assert(part.x <= x_max);
            assert(part.y <= y_max);
            assert(part.y >= y_min);
#endif

            get_idx(part.x, part.y, x_idx, y_idx);

#ifdef CHECK_ASSERT
            assert(x_idx >= 0);
            assert(x_idx < n_x);
            assert(y_idx >= 0);
            assert(y_idx < n_y);
#endif

            // Interaction with particles in the same cell
            for (int i = 0; i < part_grid[x_idx][y_idx].size(); ++i) {
                apply_force(part, *(part_grid[x_idx][y_idx].at(i)), &dmin, &davg, &navg);
#ifdef CHECK_ASSERT
                //assert(dmin > 0.4);
#endif
            }

            // South west neighboring cell
            if (x_idx > 0 && y_idx > 0) {
                for (int i = 0; i < part_grid[x_idx - 1][y_idx - 1].size(); ++i) {
                    apply_force(part, *(part_grid[x_idx - 1][y_idx - 1].at(i)), &dmin, &davg, &navg);
#ifdef CHECK_ASSERT
                    //assert(dmin > 0.4);
#endif
                }
            }

            // South neighboring cell
            if (y_idx > 0) {
                for (int i = 0; i < part_grid[x_idx][y_idx - 1].size(); ++i) {
                    apply_force(part, *(part_grid[x_idx][y_idx - 1].at(i)), &dmin, &davg, &navg);
#ifdef CHECK_ASSERT
                    //assert(dmin > 0.4);
#endif
                }
            }

            // Shout east neighboring cell
            if (x_idx < n_x - 1 && y_idx > 0) {
                for (int i = 0; i < part_grid[x_idx + 1][y_idx - 1].size(); ++i) {
                    apply_force(part, *(part_grid[x_idx + 1][y_idx - 1].at(i)), &dmin, &davg, &navg);
#ifdef CHECK_ASSERT
                    //assert(dmin > 0.4);
#endif
                }
            }

            // West neighboring cell
            if (x_idx > 0) {
                for (int i = 0; i < part_grid[x_idx - 1][y_idx].size(); ++i) {
                    apply_force(part, *(part_grid[x_idx - 1][y_idx].at(i)), &dmin, &davg, &navg);
#ifdef CHECK_ASSERT
                   //assert(dmin > 0.4);
#endif
                }
            }

            // East neighboring cell
            if (x_idx < n_x - 1) {
                for (int i = 0; i < part_grid[x_idx + 1][y_idx].size(); ++i) {
                    apply_force(part, *(part_grid[x_idx + 1][y_idx].at(i)), &dmin, &davg, &navg);
#ifdef CHECK_ASSERT
                    //assert(dmin > 0.4);
#endif
                }
            }

            // North west neighboring cell
            if (x_idx > 0 && y_idx < n_y - 1) {
                for (int i = 0; i < part_grid[x_idx - 1][y_idx + 1].size(); ++i) {
                    apply_force(part, *(part_grid[x_idx - 1][y_idx + 1].at(i)), &dmin, &davg, &navg);
#ifdef CHECK_ASSERT
                    //assert(dmin > 0.4);
#endif
                }
            }

            // North neighboring cell
            if (y_idx < n_y - 1) {
                for (int i = 0; i < part_grid[x_idx][y_idx + 1].size(); ++i) {
                    apply_force(part, *(part_grid[x_idx][y_idx + 1].at(i)), &dmin, &davg, &navg);
#ifdef CHECK_ASSERT
                    //assert(dmin > 0.4);
#endif
                }
            }

            // North east neighboring cell
            if (x_idx < n_x - 1 && y_idx < n_y - 1) {
                for (int i = 0; i < part_grid[x_idx + 1][y_idx + 1].size(); ++i) {
                    apply_force(part, *(part_grid[x_idx + 1][y_idx + 1].at(i)), &dmin, &davg, &navg);
#ifdef CHECK_ASSERT
                    //assert(dmin > 0.4);
#endif
                }
            }

            apply_forces_from_buffers(step, part);

        }

#ifdef SIM_DEBUG
        std::cout << step << ": done applying forces on " << rank << std::endl;
#endif


    }

    /**
     * Update the locations of the particles and
     * figure out which need to be sent to neighbors
     * for managing and which are in a bordering zone.
     */
    void init_locations() {

#ifdef SIM_DEBUG
        std::cout << -1 << ": mem_size for init locations " << mem.size()  << " on " << rank << std::endl;
#endif

        clear_loc_buffers();
        clear_p_buffers();

        int x_idx, y_idx;
        for (int i = 0; i < mem.size(); ++i) {

            particle_t &part = mem.at(i);

            get_idx(part.x, part.y, x_idx, y_idx);

            assert(part.x >= x_min);
            assert(part.x <= x_max);
            assert(part.y <= y_max);
            assert(part.y >= y_min);

            if(y_idx == 0 && x_idx == 0){
                // This particle is now in SW corner
                SWs_buffer.push_back(part);
            }

            if(y_idx == 0) {
                // This particle is now on S border (watch out for D. Trump...)
                Ss_buffer.push_back(part);
            }

            if(x_idx == n_x - 1 && y_idx == 0){
                // This particle is now in SE corner
                SEs_buffer.push_back(part);
            }

            if(x_idx == 0){
                // This particle is now on W border
                Ws_buffer.push_back(part);
            }

            if(x_idx == n_x - 1){
                // This particle is now on E border
                Es_buffer.push_back(part);
            }

            if(x_idx == 0 && y_idx == n_y){
                // This particle is now on NW corner
                NWs_buffer.push_back(part);
            }

            if(y_idx == n_y - 1){
                // This particle is now on North border
                Ns_buffer.push_back(part);
            }

            if(x_idx == n_x - 1 && y_idx == n_y - 1){
                // This particle is now in NE corner
                NEs_buffer.push_back(part);
            }

            part_grid[x_idx][y_idx].push_back(&mem.at(i));

        }

        /**
         * Give the location of particles in bordering zones
         * to neighbors.
         */
        comm_all_locations(-1);

    }

    /**
     * Update the locations of the particles and
     * figure out which need to be sent to neighbors
     * for managing and which are in a bordering zone.
     */
    void update_locations(int step) {

#ifdef SIM_DEBUG
        std::cout << step << ": mem_size for update locations " << mem.size()  << " on " << rank << std::endl;
#endif

        clear_loc_buffers();
        clear_p_buffers();

        int x_idx, y_idx;
        for (int i = 0; i < mem.size(); ++i) {

            particle_t &part = mem.at(i);

            move(part);

            get_idx(part.x, part.y, x_idx, y_idx);

            if(part.x >= x_min && part.x <= x_max && part.y <= y_max && part.y >= y_min){
                // Particle is still within the region
                next_mem.push_back(part);
                next_part_grid[x_idx][y_idx].push_back(&next_mem.back());

                if(y_idx == 0 && x_idx == 0){
                    // This particle is now in SW corner
                    SWs_buffer.push_back(part);
                }

                if(y_idx == 0) {
                    // This particle is now on S border (watch out for D. Trump...)
                    Ss_buffer.push_back(part);
                }

                if(x_idx == n_x - 1 && y_idx == 0){
                    // This particle is now in SE corner
                    SEs_buffer.push_back(part);
                }

                if(x_idx == 0){
                    // This particle is now on W border
                    Ws_buffer.push_back(part);
                }

                if(x_idx == n_x - 1){
                    // This particle is now on E border
                    Es_buffer.push_back(part);
                }

                if(x_idx == 0 && y_idx == n_y){
                    // This particle is now on NW corner
                    NWs_buffer.push_back(part);
                }

                if(y_idx == n_y - 1){
                    // This particle is now on North border
                    Ns_buffer.push_back(part);
                }

                if(x_idx == n_x - 1 && y_idx == n_y - 1){
                    // This particle is now in NE corner
                    NEs_buffer.push_back(part);
                }

            }else{

                // Particle is no longer within the region
                int global_idx = (int) (part.x / (size / (double) n_block_x));
                int global_idy = (int) (part.y / (size / (double) n_block_y));

                int dest = global_idx * block_stride + global_idy;

                assert(dest != rank);
                assert(dest < n_procs);
                assert(dest >= 0);

                part_send_buffers[dest].push_back(part);
            }

        }

        /**
         * Get particles from neighbors that have moved
         * within the region this processor manages
         * and add them to the memory and location grid
         */

        comm_all_particles(step);
        add_all_particles_from_buffers();

        /**
         * Give the location of particles in bordering zones
         * to neighbors.
         */
        comm_all_locations(step);

        /**
         * Swap frames
         */

        std::swap(mem, next_mem);
        std::swap(part_grid, next_part_grid);

        next_mem.clear();
        for(int i = 0; i < n_x; ++i){
            for(int j = 0; j < n_y; ++j){
                next_part_grid[i][j].clear();
            }
        }

#ifdef SIM_DEBUG
        std::cout << step << ": done updating locations on " << rank << std::endl;
#endif

    }

public:
    const double delta_x;
    const double delta_y;
    const double size;
    const int n_x;
    const int n_y;
    const int max_n_particles;
    const double x_min;
    const double x_max;
    const double y_min;
    const double y_max;

    std::vector<particle_t*>** part_grid;
    std::vector<particle_t*>** next_part_grid;

    std::vector<particle_t> mem;
    std::vector<particle_t> next_mem;

    const int rank;
    const int n_procs;
    const int block_stride;
    const int block_x;
    const int block_y;
    const int n_block_x;
    const int n_block_y;
    const double x_offset;
    const double y_offset;

    MPIComm mpicom;

    /*
     * -------------------------------------------
     *
     *              COMMUNICATION BUFFERS
     *
     * -------------------------------------------
     */

    /*
     * Send buffers for locations
     */
    std::vector<particle_t> NWs_buffer;
    std::vector<particle_t> Ns_buffer;
    std::vector<particle_t> NEs_buffer;
    std::vector<particle_t> Ws_buffer;
    std::vector<particle_t> Es_buffer;
    std::vector<particle_t> SWs_buffer;
    std::vector<particle_t> Ss_buffer;
    std::vector<particle_t> SEs_buffer;

    /*
     * Receive buffers for locations
     */
    std::vector<particle_t> NWr_buffer;
    std::vector<particle_t> Nr_buffer;
    std::vector<particle_t> NEr_buffer;
    std::vector<particle_t> Wr_buffer;
    std::vector<particle_t> Er_buffer;
    std::vector<particle_t> SWr_buffer;
    std::vector<particle_t> Sr_buffer;
    std::vector<particle_t> SEr_buffer;

    /*
     * Send/recv buffers for particles
     */
    std::vector<particle_t>* part_send_buffers;
    std::vector<particle_t>* part_recv_buffers;

    /**
     * ---------------------------------------
     *
     *              PROTECTED METHODS
     *
     * ---------------------------------------
     */

    /**
    * Look up the indices of that location.
    * IN: x, y. OUT: x_idx, y_idx
    */
    void get_idx(const double &x, const double &y, int &x_idx, int &y_idx){
        x_idx = (int) ((x - x_offset) / delta_x);
        y_idx = (int) ((y - y_offset) / delta_y);
    }

    /**
     * Clear location buffers
     */
    void clear_loc_buffers(){
        NWs_buffer.clear();
        Ns_buffer.clear();
        NEs_buffer.clear();
        Ws_buffer.clear();
        Es_buffer.clear();
        SWs_buffer.clear();
        Ss_buffer.clear();
        SEs_buffer.clear();

        NWr_buffer.clear();
        Nr_buffer.clear();
        NEr_buffer.clear();
        Wr_buffer.clear();
        Er_buffer.clear();
        SWr_buffer.clear();
        Ss_buffer.clear();
        SEs_buffer.clear();
    }

    /**
     * Clear particle buffers
     */
    void clear_p_buffers(){
        for(int i = 0; i < n_procs; ++i){
            part_send_buffers[i].clear();
            part_recv_buffers[i].clear();
        }
    }

    /**
     *
     */
    void add_all_particles_from_buffers(){
        for(int p = 0; p < n_procs; ++p){
            if(p == rank) continue;
            add_all_particles_from_buffer(part_recv_buffers[p]);
        }
    }

    /**
     * Add the content of a receive buffer to the list of particles
     */
    void add_all_particles_from_buffer(const std::vector<particle_t> &buffer){

        int x_idx, y_idx;

#ifdef SIM_DEBUG
        if(buffer.size() > 0){
            std::cout << "Adding " << buffer.size() << " particles on " << rank << std::endl;
        }
#endif

        for(int i = 0; i < buffer.size(); ++i) {

            const particle_t &part = buffer.at(i);

            get_idx(part.x, part.y, x_idx, y_idx);

#ifdef CHECK_ASSERT
            assert(part.x >= x_min);
            assert(part.x <= x_max);
            assert(part.y <= y_max);
            assert(part.y >= y_min);
            assert(x_idx >= 0);
            assert(y_idx >= 0);
            assert(x_idx < n_x);
            assert(y_idx < n_y);
#endif

            next_mem.push_back(part);
            next_part_grid[x_idx][y_idx].push_back(&next_mem.back());

            if(y_idx == 0 && x_idx == 0){
                // This particle is now in SW corner
                SWs_buffer.push_back(part);
            }

            if(y_idx == 0){
                // This particle is now on S border (watch out for D. Trump...)
                Ss_buffer.push_back(part);
            }

            if(x_idx == n_x - 1 && y_idx == 0){
                // This particle is now in SE corner
                SEs_buffer.push_back(part);
            }

            if(x_idx == 0){
                // This particle is now on W border
                Ws_buffer.push_back(part);
            }

            if(x_idx == n_x - 1){
                // This particle is now on E border
                Es_buffer.push_back(part);
            }

            if(x_idx == 0 && y_idx == n_y){
                // This particle is now on NW corner
                NWs_buffer.push_back(part);
            }

            if(y_idx == n_y - 1){
                // This particle is now on North border
                Ns_buffer.push_back(part);
            }

            if(x_idx == n_x - 1 && y_idx == n_y - 1){
                // This particle is now in NE corner
                NEs_buffer.push_back(part);
            }

        }

    }

};
