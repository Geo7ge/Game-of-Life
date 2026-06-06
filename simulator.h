#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <vector>
#include <string>

// Structură pentru a reține metadatele procesului curent
struct ProcessInfo {
    int rank;
    int num_procs;
    int global_rows;
    int global_cols;
    int local_rows;       // Liniile efective procesate de acest rank
    int start_row;        // Indexul global de start al acestui rank
    int prev_rank;        // Vecinul de deasupra (grilă toroidală)
    int next_rank;        // Vecinul de dedesubt (grilă toroidală)
};

// Alocare și inițializare
void setup_process_dimensions(int rank, int num_procs, int rows, int cols, ProcessInfo& info);
void initialize_grid(std::vector<int>& grid, const ProcessInfo& info, bool random_init = true);

// Comunicație Halo Exchange
void exchange_halo_1d(std::vector<int>& grid, const ProcessInfo& info);

// Logica de evoluție a Game of Life
int count_neighbors(const std::vector<int>& grid, int r, int c, const ProcessInfo& info);
void evolve_local_grid(const std::vector<int>& current_grid, std::vector<int>& next_grid, const ProcessInfo& info);

// Validare și output
void save_grid_pgm(const std::vector<int>& grid, const ProcessInfo& info, const std::string& filename);

#endif // SIMULATOR_H