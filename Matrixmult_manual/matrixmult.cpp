#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <ctime>
#include <cstdlib> 

//SIMPLE PROGRAM COMPERING COMPUTE TIMES OF THREADED AND SEQUENTIAL 
//MATRIX MULTIPLICATION WITH MANUAL MEMORY MANAGEMENT.
//COMPARED ARE ALSO STRAIGHTFORWARD MULTIPLICATION METHOD, AND MORE MEMORY
//ACCESS EFICIENT ONE, INVOLVING TRANSPOSITION.

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

void fill_zeros(int** M, int N){
    for (int i=0; i<N; i++){
        for (int j=0; j <N; j++){
            M[i][j] =0;
        }
    }
}


void fill_matrix(int** matrix, int N) {
    std::srand(std::time(nullptr)); 
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            matrix[i][j] = std::rand() % 100;
        }
    }
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


int** transpose(int** M2, int N) {
    int** M2T = new int*[N]; 
    for (int i = 0; i < N; ++i) {
        M2T[i] = new int[N]; 
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            M2T[j][i] = M2[i][j];  
        }
    }

    return M2T;  
}



void matrix_mult_transpose(int** M1, int** M2T, int** M_mult, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                M_mult[i][j] += M1[i][k] * M2T[j][k];
            }
        }
    }
}

void thread_transposed_matrix_mult(int tid, int** M1, int** M2T, int** M_mult, int N, int it) {    
    int start = it  * tid;
    int fin = start + it;
    for(int i=start; i<fin; i++){
        for(int j= 0; j<N; j++){
            for(int k =0; k<N; k++){
                M_mult[i][j] += M1[i][k] * M2T[j][k];
            }
        }

    }
}

int main() {
    
    for (int N = 240; N <= 1200; N = N + 240) {
        int num_threads = 12;
        int it; // iterations per thread

        int** matrix1 = allocate_matrix(N);
        int** matrix2 = allocate_matrix(N);
        int** M_mult = allocate_matrix(N);

        fill_matrix(matrix1, N);
        fill_matrix(matrix2, N);

        auto start_seq = std::chrono::high_resolution_clock::now();
        matrix_mult(matrix1, matrix2, M_mult, N);
        auto end_seq = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> seq_time = end_seq - start_seq;

        fill_zeros(M_mult, N);

       
        int** matrix2_T = transpose(matrix2, N);
        auto start_opt = std::chrono::high_resolution_clock::now();
        matrix_mult_transpose(matrix1, matrix2_T, M_mult, N);
        auto end_opt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> opt_time = end_opt - start_opt;

        fill_zeros(M_mult, N);

        for (num_threads = 2; num_threads <= 12; num_threads += 2) {
            fill_zeros(M_mult, N);
            
            it = N / num_threads;

            auto start_th = std::chrono::high_resolution_clock::now();
            std::vector<std::thread> thread_vec;
            for (int i = 0; i < num_threads; ++i) {
                thread_vec.push_back(std::thread(thread_matrix_mult, i, matrix1, matrix2, M_mult, N, it));
            }

            for (int i = 0; i < num_threads; ++i) {
                thread_vec[i].join();
            }
            auto end_th = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> th_time = end_th - start_th;


            fill_zeros(M_mult, N);

            auto start_th_T = std::chrono::high_resolution_clock::now();
            std::vector<std::thread> thread_vec_T;
            for (int i = 0; i < num_threads; ++i) {
                thread_vec_T.push_back(std::thread(thread_transposed_matrix_mult, i, matrix1, matrix2_T, M_mult, N, it));
            }

            for (int i = 0; i < num_threads; ++i) {
                thread_vec_T[i].join();
            }


            auto end_th_T = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> th_time_T = end_th_T - start_th_T;

            std::cout << "N = " << N << ", Threads: " << num_threads 
                      << ", Sequential: " << seq_time.count() << " s"
                      << ", Sequential (transposed): " << opt_time.count() << " s"
                      << ", Threaded: " << th_time.count() << " s"
                      << ", Threaded (transposed): "<< th_time_T.count() << " s\n";
        }

        free_matrix(matrix1, N);
        free_matrix(matrix2, N);
        free_matrix(matrix2_T, N);  
        free_matrix(M_mult, N);

        std::cout<<"\n"; 
    }
    
    return 0;
}
