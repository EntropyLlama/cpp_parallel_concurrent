#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <mutex>
#include <omp.h>

//PROGRAM THAT TRAVELS THROUGH LABYRINTH IN PARALLEL AND WRITES TID IN THE PATHS. LABYRINTH IS REPRESENTED BY MATRIX
//WHERE -1 ARE WALLS AND 0 ARE PATHS

// -fopenmp

std::vector<std::vector<int>> labyrinth;

//std::mutex mtx[100][100];
omp_lock_t mtx[100][100];

std::vector<std::vector<int>> readFromFile(const std::string& filename) {
    std::ifstream in_matrix_file(filename);
    std::vector<std::vector<int>> matrix;

    if (!in_matrix_file.is_open()) {
        std::cout << "ERROR: Failed to open the file\n";
        return matrix;
    }

    std::string line;
    while (std::getline(in_matrix_file, line)) {
        std::vector<int> row;
        for (char ch : line) {
            if (ch == '0' || ch == '1') {
                row.push_back(ch - '0');
            }
        }
        matrix.push_back(row);
    }
    return matrix;
}

void printMatrix(const std::vector<std::vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }
}
//in case imput matrix walls are represented by 1
void makeWallsNegative(std::vector<std::vector<int>>& matrix) {
    for (auto& row : matrix) {
        for (auto& value : row) {
            if (value == 1) value = -1;
        }
    }
}
//traversal function - checks 4 directions - makes new threads when it finds more than 1
void labyrinthTraversal_thread(int tid, int row, int col) {
    
    bool path_found;
    omp_set_lock(&mtx[row][col]);
    if(labyrinth[row][col]==0){
        labyrinth[row][col] = tid;
        path_found = true;
    }
    else{
        path_found = false;
    }
    omp_unset_lock(&mtx[row][col]);

    std::vector<std::pair<int, int>> directions = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};

    int new_threads_created = 0;
    int new_row;
    int new_col;
 
    while (path_found) {
        path_found = false;
        if (row >= 0 && row < labyrinth.size() && col >= 0 && col < labyrinth[0].size()) {
            for (int i = 0; i < directions.size(); ++i) {
            
                int checked_row = row + directions[i].first;
                int checked_col = col + directions[i].second;
                omp_set_lock(&mtx[checked_row][checked_col]);
                if (labyrinth[checked_row][checked_col] == 0) {
                    if (!path_found) {
                        labyrinth[checked_row][checked_col] = tid;
                        
                        path_found = true;
                        new_row = checked_row;
                        new_col = checked_col;
        
                    }
                    else 
                    {   
                        #pragma omp task firstprivate(checked_row, checked_col, tid)
                        {
                        labyrinthTraversal_thread(tid + 1 + new_threads_created, checked_row, checked_col);
                        }
                        new_threads_created++;
                    }
                }
                omp_unset_lock(&mtx[checked_row][checked_col]);
            }
        }
        row = new_row;
        col = new_col;
    }
    #pragma omp taskwait
}


void saveToPPM(const std::vector<std::vector<int>>& labyrinth, const std::string& filename) {
    std::ofstream ppm_file(filename, std::ios::binary);
    int scale_factor = 9;
    int width = labyrinth[0].size() * scale_factor;
    int height = labyrinth.size() * scale_factor;

    ppm_file << "P3\n" << width << " " << height << "\n255\n";

    for (const auto& row : labyrinth) {
        for (int i = 0; i < scale_factor; ++i) {
            for (const auto& cell : row) {
                for (int j = 0; j < scale_factor; ++j) {
                    if (cell == -1) ppm_file << "0 0 0 ";
                    else if (cell == 0) ppm_file << "255 255 255 ";
                    else {
                       switch (cell % 12) { 
                            case 1: ppm_file << "255 0 0 "; break;       // Red
                            case 2: ppm_file << "0 255 0 "; break;       // Green
                            case 3: ppm_file << "0 0 255 "; break;       // Blue
                            case 4: ppm_file << "255 255 0 "; break;     // Yellow
                            case 5: ppm_file << "255 0 255 "; break;     // Magenta
                            case 6: ppm_file << "0 255 255 "; break;     // Cyan
                            case 7: ppm_file << "255 165 0 "; break;     // Orange
                            case 8: ppm_file << "128 0 128 "; break;     // Purple
                            case 9: ppm_file << "0 128 0 "; break;       // Dark Green
                            case 10: ppm_file << "128 128 128 "; break;  // Gray
                            case 11: ppm_file << "255 192 203 "; break;  // Pink
                            case 0: ppm_file << "255 255 255 "; break;   // White 
                        }

                    }
                }
            }
            ppm_file << "\n";
        }
    }
   
}


int main() {
    
    std::string filename = "labyrinth.txt";
    labyrinth = readFromFile(filename);
    makeWallsNegative(labyrinth);

    //start at [1, 1]
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) { omp_init_lock(&mtx[i][j]); }
    }

    
    auto start = std::chrono::high_resolution_clock::now();
    int tid = 1;
    #pragma omp parallel num_threads(12)
    {
        #pragma omp single
        {
            labyrinthTraversal_thread(tid, 1, 1);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << elapsed.count() << " seconds\n";


    #pragma omp taskwait

    printMatrix(labyrinth);
    saveToPPM(labyrinth, "traversed_labyrinth.ppm");

    return 0;
}
