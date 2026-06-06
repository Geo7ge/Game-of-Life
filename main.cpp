#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include "simulator.h"

int main(int argc, char* argv[]) {
    // Dimensiuni implicite pentru grilă și număr de generații
    int rows = 1000;
    int cols = 1000;
    int generations = 100;
    bool random_init = true;

    // Citirea argumentelor din linie de comandă (dacă există)
    if (argc >= 4) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
        generations = std::atoi(argv[3]);
        if (argc >= 5) {
            random_init = (std::atoi(argv[4]) != 0);
        }
    }

    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0) {
        std::cout << "=== Simulator Parallel Conway's Game of Life ===" << std::endl;
        std::cout << "Dimensiune grila globala: " << rows << " x " << cols << std::endl;
        std::cout << "Numar generatii: " << generations << std::endl;
        std::cout << "Numar procese MPI implicate: " << num_procs << std::endl;
    }

    // Configurare geometrie și alocare vectori
    ProcessInfo info;
    setup_process_dimensions(rank, num_procs, rows, cols, info);

    std::vector<int> current_grid;
    std::vector<int> next_grid;

    initialize_grid(current_grid, info, random_init);
    initialize_grid(next_grid, info, false); // Initializează bufferul secundar cu 0

    // Salvare stare inițială (Generația 0) pentru debug/vizualizare
    if (!random_init || (rows <= 100 && cols <= 100)) {
        save_grid_pgm(current_grid, info, "generation_0.pgm");
    }

    // Sincronizare înainte de pornirea cronometrului
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Bucla principală a simulării
    for (int gen = 1; gen <= generations; ++gen) {
        // Pasul 1: Halo Exchange (schimb de frontiere între vecini)
        exchange_halo_1d(current_grid, info);

        // Pasul 2: Calcul local pentru generația următoare
        evolve_local_grid(current_grid, next_grid, info);

        // Pasul 3: Swap de pointeri / vectori pentru runda următoare
        current_grid.swap(next_grid);
    }

    // Sincronizare finală și oprire cronometru
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // Colectarea metricilor de performanță pe Rank 0
    double max_elapsed_time;
    MPI_Reduce(&elapsed_time, &max_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\nSimulare finalizata cu succes!" << std::endl;
        std::cout << "Timp total executie paralel: " << max_elapsed_time << " secunde." << std::endl;
        std::cout << "Performanta: " << (double(rows) * cols * generations) / (max_elapsed_time * 1e6) 
                  << " milioane celule actualizate/secunda." << std::endl;
    }

    // Salvare stare finală
    if (!random_init || (rows <= 100 && cols <= 100)) {
        save_grid_pgm(current_grid, info, "generation_final.pgm");
    }

    MPI_Finalize();
    return 0;
}