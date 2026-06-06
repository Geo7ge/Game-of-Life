#include "simulator.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

void setup_process_dimensions(int rank, int num_procs, int rows, int cols, ProcessInfo& info) {
    info.rank = rank;
    info.num_procs = num_procs;
    info.global_rows = rows;
    info.global_cols = cols;

    // Împărțirea uniformă a liniilor (Descompunere 1D)
    info.local_rows = rows / num_procs;
    int remainder = rows % num_procs;
    
    // Distribuirea restului de linii către primele procese
    if (rank < remainder) {
        info.local_rows++;
        info.start_row = rank * info.local_rows;
    } else {
        info.start_row = rank * info.local_rows + remainder;
    }

    // Topologie toroidală pe verticală (wrap-around)
    info.prev_rank = (rank - 1 + num_procs) % num_procs;
    info.next_rank = (rank + 1) % num_procs;
}

void initialize_grid(std::vector<int>& grid, const ProcessInfo& info, bool random_init) {
    // Dimensiunea totală vector local: (linii_locale + 2 pentru halo) * coloane_globale
    int total_local_cells = (info.local_rows + 2) * info.global_cols;
    grid.assign(total_local_cells, 0);

    if (random_init) {
        // Inițializare pseudo-random bazată pe rank pentru reproductibilitate granulară
        std::srand(1337 + info.rank); 
        for (int r = 1; r <= info.local_rows; ++r) {
            for (int c = 0; c < info.global_cols; ++c) {
                grid[r * info.global_cols + c] = (std::rand() % 100 < 20) ? 1 : 0; // 20% șanse celulă vie
            }
        }
    } else {
        // Exemplu: Inițializare cu un "Glider" dacă spațiul global o permite
        if (info.start_row <= 2 && info.start_row + info.local_rows > 2 && info.global_cols >= 3) {
            int local_r2 = 2 - info.start_row + 1; // mapare în index local cu halo
            grid[local_r2 * info.global_cols + 1] = 1;
            grid[(local_r2 + 1) * info.global_cols + 2] = 1;
            grid[local_r2 * info.global_cols + 3] = 1;
            grid[(local_r2 - 1) * info.global_cols + 3] = 1;
            grid[(local_r2 + 1) * info.global_cols + 3] = 1;
        }
    }
}

void exchange_halo_1d(std::vector<int>& grid, const ProcessInfo& info) {
    MPI_Request requests[4];
    
    // Pointeri către liniile de trimis/primit
    int* send_top_row = &grid[1 * info.global_cols];
    int* send_bottom_row = &grid[info.local_rows * info.global_cols];
    int* recv_top_halo = &grid[0 * info.global_cols];
    int* recv_bottom_halo = &grid[(info.local_rows + 1) * info.global_cols];

    // Dacă rulăm pe un singur proces, facem wrap-around local direct în memorie
    if (info.num_procs == 1) {
        for (int c = 0; c < info.global_cols; ++c) {
            recv_top_halo[c] = send_bottom_row[c];
            recv_bottom_halo[c] = send_top_row[c];
        }
        return;
    }

    // Comunicații non-blocking pentru evitarea deadlock-ului
    MPI_Irecv(recv_top_halo, info.global_cols, MPI_INT, info.prev_rank, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(recv_bottom_halo, info.global_cols, MPI_INT, info.next_rank, 1, MPI_COMM_WORLD, &requests[1]);
    
    MPI_Isend(send_top_row, info.global_cols, MPI_INT, info.prev_rank, 1, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(send_bottom_row, info.global_cols, MPI_INT, info.next_rank, 0, MPI_COMM_WORLD, &requests[3]);

    // Așteptăm completarea tuturor transferurilor
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
}

int count_neighbors(const std::vector<int>& grid, int r, int c, const ProcessInfo& info) {
    int count = 0;
    int cols = info.global_cols;

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i == 0 && j == 0) continue;

            int neighbor_r = r + i;
            // Wrap-around pe orizontală (stânga-dreapta toroidat local)
            int neighbor_c = (c + j + cols) % cols; 

            // Pe verticală ne bazăm pe liniile din halo (0 și local_rows + 1) populate corect de MPI
            count += grid[neighbor_r * cols + neighbor_c];
        }
    }
    return count;
}

void evolve_local_grid(const std::vector<int>& current_grid, std::vector<int>& next_grid, const ProcessInfo& info) {
    int cols = info.global_cols;

    for (int r = 1; r <= info.local_rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int live_neighbors = count_neighbors(current_grid, r, c, info);
            int current_idx = r * cols + c;

            if (current_grid[current_idx] == 1) {
                // Regulile 1 și 2 (Subpopulare și Supraviețuire)
                if (live_neighbors < 2 || live_neighbors > 3) {
                    next_grid[current_idx] = 0; // Moare
                } else {
                    next_grid[current_idx] = 1; // Supraviețuiește
                }
            } else {
                // Regula de reproducere (exact 3 vecini vii)
                if (live_neighbors == 3) {
                    next_grid[current_idx] = 1; // Învie
                } else {
                    next_grid[current_idx] = 0;
                }
            }
        }
    }
}

void save_grid_pgm(const std::vector<int>& grid, const ProcessInfo& info, const std::string& filename) {
    // Procesul 0 colectează datele de la toate procesele pentru a scrie fișierul imagine ordonat
    int cols = info.global_cols;
    
    if (info.rank == 0) {
        std::ofstream out(filename);
        if (!out.is_open()) return;

        // Header format PGM binar sau ASCII (folosim ASCII P2 pentru simplitate și debug vizual facil)
        out << "P2\n" << cols << " " << info.global_rows << "\n255\n";

        // Scriem porțiunea locală a procesului 0
        for (int r = 1; r <= info.local_rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                out << (grid[r * cols + c] ? "0 " : "255 "); // Negru pentru viu, alb pentru mort
            }
            out << "\n";
        }

        // Primim pe rând liniile de la celelalte procese
        for (int p = 1; p < info.num_procs; ++p) {
            // Aflăm câte linii trimite procesul p
            int p_local_rows = info.global_rows / info.num_procs;
            int remainder = info.global_rows % info.num_procs;
            if (p < remainder) p_local_rows++;

            std::vector<int> buffer(p_local_rows * cols);
            MPI_Recv(buffer.data(), p_local_rows * cols, MPI_INT, p, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < p_local_rows * cols; ++i) {
                out << (buffer[i] ? "0 " : "255 ");
                if ((i + 1) % cols == 0) out << "\n";
            }
        }
        out.close();
    } else {
        // Celelalte procese trimit doar zona lor utilă (fără liniile de halo)
        const int* data_to_send = &grid[1 * cols];
        MPI_Send(data_to_send, info.local_rows * cols, MPI_INT, 0, 100, MPI_COMM_WORLD);
    }
}