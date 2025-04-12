#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
//SIMPLE PROGRAM FOR PARALLEL MATRIX MULTIPLICATION USING OMP

int** allocate_matrix(int N) {
    int** matrix = new int*[N];
    for (int i = 0; i < N; i++) {
        matrix[i] = new int[N];
    }
    return matrix;
}


void free_matrix(int** matrix, int N) {
    for (int i = 0; i < N; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}


void fill_zeros(int** M, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            M[i][j] = 0;
        }
    }
}


void fill_matrix(int** matrix, int N) {
    std::srand(std::time(nullptr));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = std::rand() % 100;
        }
    }
}


int** transpose(int** M, int N) {
    int** M_T = allocate_matrix(N);
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            M_T[j][i] = M[i][j];
        }
    }
    return M_T;
}

void matrix_mult(int** M1, int** M2, int** M_mult, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                M_mult[i][j] += M1[i][k] * M2[k][j];
            }
        }
    }
}


void thread_matrix_mult(int tid, int** M1, int** M2, int** M_mult, int N, int it) {
    int start = it  * tid;
    int fin = start + it;
    for(int i=start; i<fin; i++){
        for(int j= 0; j<N; j++){
            for(int k =0; k<N; k++){
                M_mult[i][j] += M1[i][k] * M2[k][j];
            }
        }

    }
}


void matrix_mult_omp(int** A, int** B, int** C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}


void matrix_mult_transpose_omp(int** A, int** B_T, int** C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B_T[j][k];
            }
        }
    }
}


// void tiled_matrix_mult_omp(int** A, int** B, int** C, int N, int tile_size) {
//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < N; i += tile_size) {
//         for (int j = 0; j < N; j += tile_size) {
//             for (int k = 0; k < N; k += tile_size) {
//                 for (int ii = i; ii < std::min(i + tile_size, N); ii++) {
//                     for (int jj = j; jj < std::min(j + tile_size, N); jj++) {
//                         for (int kk = k; kk < std::min(k + tile_size, N); kk++) {
//                             C[ii][jj] += A[ii][kk] * B[kk][jj];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }
void tiled_matrix_mult_omp(int** A, int** B, int** C, int N, int tile_size) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += tile_size) {
        for (int j = 0; j < N; j += tile_size) {
            for (int k = 0; k < N; k += tile_size) {
                for (int ii = i; ii < std::min(i + tile_size, N); ii++) {
                    for (int jj = j; jj < std::min(j + tile_size, N); jj++) {
                        int sum = 0;  
                        for (int kk = k; kk < std::min(k + tile_size, N); kk++) {
                            sum += A[ii][kk] * B[kk][jj];
                        }
                        C[ii][jj] += sum;  
                    }
                }
            }
        }
    }
}

void tiled_matrix_mult_transpose_omp(int** A, int** B_T, int** C, int N, int tile_size) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += tile_size) {
        for (int j = 0; j < N; j += tile_size) {
            for (int k = 0; k < N; k += tile_size) {
                for (int ii = i; ii < std::min(i + tile_size, N); ii++) {
                    for (int jj = j; jj < std::min(j + tile_size, N); jj++) {
                        int sum = 0;  
                        for (int kk = k; kk < std::min(k + tile_size, N); kk++) {
                            sum += A[ii][kk] * B_T[jj][kk];
                        }
                        C[ii][jj] += sum;  
                    }
                }
            }
        }
    }
}



#include <fstream>
#include <sstream>

int main() {
    const int tile_sizes[] = {4, 8, 16, 32, 64, 128, 256}; 

    std::ofstream file("matrix_multiplication_results.csv");
    file << "Method,Tile Size,Time (seconds)\n";

    for (int N = 1024; N <= 4096; N += 1024) {
        std::cout << "Matrix size N = " << N << "\n";
        file <<"Matrix size: " << N <<"\n";
        int** A = allocate_matrix(N);
        int** B = allocate_matrix(N);
        int** C = allocate_matrix(N);

        fill_matrix(A, N);
        fill_matrix(B, N);

        fill_zeros(C, N);
        double start_time = omp_get_wtime();
        matrix_mult_omp(A, B, C, N);
        double end_time = omp_get_wtime();
        double naive_time = end_time - start_time;

        

        std::cout << "Naive Multiplication Time: " << naive_time << " seconds\n";
        file << N << ",Naive,NA," << naive_time << "\n";

        fill_zeros(C, N);
        int** B_T = transpose(B, N);
        start_time = omp_get_wtime();
        matrix_mult_transpose_omp(A, B_T, C, N);
        end_time = omp_get_wtime();
        double transpose_time = end_time - start_time;
        std::cout << "Transposed Multiplication Time: " << transpose_time << " seconds\n";
        file << ",Transposed,NA," << transpose_time << "\n";

        for (int tile_size : tile_sizes) {
            fill_zeros(C, N);
            start_time = omp_get_wtime();
            tiled_matrix_mult_omp(A, B, C, N, tile_size);
            end_time = omp_get_wtime();
            double tiled_time = end_time - start_time;
            std::cout << "Tiled Multiplication Time (Tile Size " << tile_size << "x" << tile_size << "): " << tiled_time << " seconds\n";
            file << ",Tiled," << tile_size << "," << tiled_time << "\n";
        }
        file << "\n";
        for (int tile_size : tile_sizes) {
            fill_zeros(C, N);
            start_time = omp_get_wtime();
            tiled_matrix_mult_transpose_omp(A, B_T, C, N, tile_size);
            end_time = omp_get_wtime();
            double tiled_transpose_time = end_time - start_time;
            std::cout << "Transposed Tiled Multiplication Time(Tile Size " << tile_size << "x" << tile_size << "): " << tiled_transpose_time << " seconds\n";
            file << ",Tiled Transposed," << tile_size << "," << tiled_transpose_time << "\n";
        }

        free_matrix(B_T, N);
        free_matrix(A, N);
        free_matrix(B, N);
        free_matrix(C, N);

        std::cout << "\n";
        file << "\n";
    }

    file.close();
    return 0;
}
