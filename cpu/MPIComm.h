//
// Created by francois.belletti on 3/4/16.
//

#pragma once

#include "common.h"
#include <stdlib.h>
#include <vector>
#include <assert.h>
#include <iostream>
#include <mpi.h>

//#define COMM_DEBUG


class MPIComm{

    MPI_Datatype PARTICLE;
    const int rank;

public:
    MPIComm(int _rank):
    rank(_rank)
    {
        MPI_Type_contiguous(6, MPI_DOUBLE, &PARTICLE);
        MPI_Type_commit(&PARTICLE);
    }

    /*
     * Send and receive the content of a buffer
     */
    void exchange_buffers(int target, int source,
                          bool should_send,
                          bool should_recv,
                          int &msg_idx,
                          std::vector<particle_t> &send_buffer,
                          std::vector<particle_t> &recv_buffer,
                          std::string comment = ""){

        int n_reqs = should_send + should_recv;

        MPI_Request reqs[n_reqs];
        MPI_Status  status[n_reqs];

#ifdef COMM_DEBUG
        if(should_send) {
            std::cout << "\tO sending " << send_buffer.size() << " " << comment << " to " << target << " on " << rank << std::endl;
        }
        if(should_recv) {
            std::cout << "\tO receiving " << comment << " from " << source << " on " << rank << std::endl;
        }
#endif

        int n_to_send = send_buffer.size();
        int n_to_recv = -1;

        int msg_tag = ++msg_idx;
        // Send info about size to NW neighbor
        if (should_send) {
            MPI_Isend(&(n_to_send), 1, MPI_INT, target, msg_tag, MPI_COMM_WORLD, reqs);
        }
        // Receive info about size from the SE neighbor
        if (should_recv) {
            MPI_Irecv(&(n_to_recv), 1, MPI_INT, source, msg_tag, MPI_COMM_WORLD, reqs + n_reqs - 1);
        }
        //std::cout << "O Exchanging size data on " << rank << std::endl;
        MPI_Waitall(n_reqs, reqs, status);
        //std::cout << "X Exchanged size data on " << rank << std::endl;

        if (should_recv) {
            assert(n_to_recv >= 0);
        }

        msg_tag = ++msg_idx;
        // Send particles
        if (should_send) {
            MPI_Isend(send_buffer.data(), n_to_send, PARTICLE, target, msg_tag, MPI_COMM_WORLD, reqs);
        }
        // Receive particles
        if (should_recv) {
            recv_buffer.clear();
            recv_buffer.resize(n_to_recv);
            MPI_Irecv(recv_buffer.data(), n_to_recv, PARTICLE, source, msg_tag, MPI_COMM_WORLD, reqs + n_reqs - 1);
        }
        //std::cout << "O Exchanging particle data on " << rank << std::endl;
        MPI_Waitall(n_reqs, reqs, status);
        //std::cout << "X Exchanged particle data on " << rank << std::endl;

#ifdef COMM_DEBUG
        if(should_send) {
            std::cout << "\tO sent " << send_buffer.size() << " " << comment << " to " << target << " on " << rank << std::endl;
        }
        if(should_recv) {
            std::cout << "\tX received " << n_to_recv << " " << comment << " from " << source << " on " << rank << std::endl;
        }
#endif

    }

    /**
     * Particle buffer utils
     */
    void send_buffer_size(int to, int msg_tag, const std::vector<particle_t> &buff, MPI_Request &req) {

        int buff_size = buff.size();
        MPI_Isend(&(buff_size), 1, MPI_INT, to, msg_tag, MPI_COMM_WORLD, &req);

    }

    void send_buffer(int to, int msg_tag, std::vector<particle_t> &buff, MPI_Request &req) {

        MPI_Isend(buff.data(), buff.size(), PARTICLE, to, msg_tag, MPI_COMM_WORLD, &req);

    }

    void recv_buffer_size(int from, int msg_tag, int &buffsize, MPI_Request &req) {

        MPI_Irecv(&buffsize, 1, MPI_INT, from, msg_tag, MPI_COMM_WORLD, &req);

    }

    void recv_buffer(int from, int msg_tag, std::vector<particle_t> &buff, MPI_Request &req) {

        MPI_Irecv(buff.data(), buff.size(), PARTICLE, from, msg_tag, MPI_COMM_WORLD, &req);

    }

};