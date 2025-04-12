#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <omp.h>

//PROGRAM GENERATING ULAM SPIRAL IN DIFFERENT WAYS, COMPERING TIMES

bool isPrime(unsigned u) {
    if (u < 4) return u > 1;
    if (!(u % 2) || !(u % 3)) return false;

    unsigned q = static_cast<unsigned>(std::sqrt(static_cast<long double>(u))), c = 5;
    while (c <= q) {
        if (!(u % c) || !(u % (c + 2))) return false;
        c += 6;
    }
    return true;
}


int spiralValue(int x, int y, int n) {
    int c = n / 2;
    int layer = std::max(std::abs(x - c), std::abs(y - c));
    int max_val = (2 * layer + 1) * (2 * layer + 1);

    int offset = 0;
    if (y == c + layer) {
        offset = c + layer - x;
    } else if (x == c - layer) {
        offset = (2 * layer) + (c + layer - y);
    } else if (y == c - layer) {
        offset = (4 * layer) + (x - (c - layer));
    } else if (x == c + layer) {
        offset = (6 * layer) + (y - (c - layer));
    }
    return max_val - offset;
}


void generateUlamSpiralSequential(std::vector<int>& spiral, unsigned N) {
    for (unsigned posY = 0; posY < N; posY++) {
        for (unsigned posX = 0; posX < N; posX++) {
            unsigned value = spiralValue(posX, posY, N);
            spiral[posY * N + posX] = isPrime(value) ? 0 : value;
        }
    }
}


void UlamHorizontal(std::vector<int>& spiral, unsigned N) {
    #pragma omp parallel for schedule(static)
    for (unsigned posY = 0; posY < N; posY++) {
        for (unsigned posX = 0; posX < N; posX++) {
            unsigned value = spiralValue(posX, posY, N);
            spiral[posY * N + posX] = isPrime(value) ? 0 : value;
        }
    }
}


void Ulam2x2(std::vector<int>& spiral, unsigned N) {
    int tileSize = N / 2;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int blockY = 0; blockY < 2; blockY++) {
        for (int blockX = 0; blockX < 2; blockX++) {
            for (unsigned posY = blockY * tileSize; posY < (blockY + 1) * tileSize; posY++) {
                for (unsigned posX = blockX * tileSize; posX < (blockX + 1) * tileSize; posX++) {
                    unsigned value = spiralValue(posX, posY, N);
                    spiral[posY * N + posX] = isPrime(value) ? 0 : value;
                }
            }
        }
    }
}


void Ulam4x4(std::vector<int>& spiral, unsigned N) {
    int tileSize = N / 4;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int blockY = 0; blockY < 4; blockY++) {
        for (int blockX = 0; blockX < 4; blockX++) {
            for (unsigned posY = blockY * tileSize; posY < (blockY + 1) * tileSize; posY++) {
                for (unsigned posX = blockX * tileSize; posX < (blockX + 1) * tileSize; posX++) {
                    unsigned value = spiralValue(posX, posY, N);
                    spiral[posY * N + posX] = isPrime(value) ? 0 : value;
                }
            }
        }
    }
}


void displayNumbers(const std::vector<int>& spiral, unsigned N) {
    unsigned ct = 0, width = static_cast<unsigned>(std::log10(N * N)) + 1;
    for (auto i : spiral) {
        if (i) std::cout << std::setw(width) << i << " ";
        else std::cout << std::string(width, '*') << " ";
        if (++ct >= N) {
            std::cout << "\n";
            ct = 0;
        }
    }
    std::cout << "\n\n";
}

#include <fstream>

#include <string>


void saveVectorToPPM(const std::string& filename, const std::vector<int>& vec, int N) {

    if (vec.size() != N * N) {
        std::cerr << "Error: Vector size does not match the matrix size N x N" << std::endl;
        return;
    }


    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing" << std::endl;
        return;
    }


    file << "P3\n";
    file << N << " " << N << "\n";
    file << "255\n";


    for (int i = 0; i < N * N; ++i) {
        if (vec[i] == 0) {

            file << "255 255 255 ";
        } else {

            file << "0 0 0 ";
        }


        if ((i + 1) % N == 0) {
            file << "\n";
        }
    }


    file.close();
    std::cout << "PPM file saved successfully: " << filename << std::endl;
}




int main() {
    unsigned N = 4096; // Size of the spiral
    std::vector<int> spiral(N * N, -1);

    std::cout << "Generating Ulam Spiral of size " << N << "x" << N << "\n";


    auto start = std::chrono::high_resolution_clock::now();
    generateUlamSpiralSequential(spiral, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Sequential: " << elapsed.count() << " seconds\n";


    spiral.assign(N * N, -1);
    start = std::chrono::high_resolution_clock::now();
    UlamHorizontal(spiral, N);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Parallel Horizontal: " << elapsed.count() << " seconds\n";


    spiral.assign(N * N, -1);
    start = std::chrono::high_resolution_clock::now();
    Ulam2x2(spiral, N);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Parallel 2x2 Blocks: " << elapsed.count() << " seconds\n";


    spiral.assign(N * N, -1);
    start = std::chrono::high_resolution_clock::now();
    Ulam4x4(spiral, N);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Parallel 4x4 Blocks: " << elapsed.count() << " seconds\n";

    saveVectorToPPM("spiral.ppm",spiral, N);

    return 0;
}
