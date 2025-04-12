#include <stdio.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <omp.h>
#include <fstream>
// -lgomp -fopenmp -O3
//TESTING OMP ON MANDELBROT
const int iXmax = 800;
const int iYmax = 800;
const int MaxColorComponentValue = 255;
const int IterationMax = 200;
const double EscapeRadius = 2;
double ER2 = EscapeRadius * EscapeRadius;

#include <fstream>

std::string filename = "data.csv";

#include <sstream>

// Function to save data to CSV file
void saveDataToCSV(const std::string& filename, const std::string& schedule_type, int chunk_size, int num_threads, double elapsed_time, const int* total_iterations, int max_threads) {
    std::ofstream file;
    file.open(filename, std::ios::app); // Open in append mode

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing.\n";
        return;
    }

    // Write header if the file is empty
    static bool is_header_written = false;
    if (!is_header_written) {
        file << "Schedule,ChunkSize,Threads,ElapsedTime,ThreadID,Iterations\n";
        is_header_written = true;
    }

    for (int i = 0; i < max_threads; ++i) {
        file << schedule_type << "," << chunk_size << "," << num_threads << "," << elapsed_time << "," << i << "," << total_iterations[i] << "\n";
    }

    file.close();
}


unsigned char thread_colors[12][3] = {
    {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0},
    {255, 0, 255}, {0, 255, 255}, {128, 0, 0}, {0, 128, 0},
    {0, 0, 128}, {128, 128, 0}, {128, 0, 128}, {0, 128, 128}
};

void computeMandelbrot(int width, int height, int num_threads, const std::string& schedule_type, int chunk_size) {
    std::vector<unsigned char> pixel_vec(3 * width * height);
    int total_iterations[12] = {0}; 
    omp_set_num_threads(num_threads); 

    double PixelWidth = (1.5 - (-2.5)) / width;
    double PixelHeight = (2.0 - (-2.0)) / height;

    double start_time = omp_get_wtime();

#pragma omp parallel for schedule(runtime) shared(pixel_vec, PixelWidth, PixelHeight, total_iterations)
    for (int iY = 0; iY < height; ++iY) {
        
        int thread_id = omp_get_thread_num();
        //total_iterations[thread_id]+=1;
        double Cy = -2.0 + iY * PixelHeight;
        if (fabs(Cy) < PixelHeight / 2) Cy = 0.0;

        for (int iX = 0; iX < width; ++iX) {
            double Cx = -2.5 + iX * PixelWidth;
            double Zx = 0.0;
            double Zy = 0.0;
            double Zx2 = Zx * Zx;
            double Zy2 = Zy * Zy;
            int Iteration;

            for (Iteration = 0; Iteration < IterationMax && ((Zx2 + Zy2) < ER2); Iteration++) {
                Zy = 2 * Zx * Zy + Cy;
                Zx = Zx2 - Zy2 + Cx;
                Zx2 = Zx * Zx;
                Zy2 = Zy * Zy;
            }

            int index = (iY * width + iX) * 3;
            total_iterations[thread_id] += Iteration;

            if (Iteration == IterationMax) {
                pixel_vec[index] = 0;
                pixel_vec[index + 1] = 0;
                pixel_vec[index + 2] = 0;
            } else {
                pixel_vec[index] = thread_colors[thread_id % 12][0];
                pixel_vec[index + 1] = thread_colors[thread_id % 12][1];
                pixel_vec[index + 2] = thread_colors[thread_id % 12][2];
            }
        }
    }

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    std::cout << "Threads: " << num_threads << ", Schedule: " << schedule_type
              << ", Chunk size: " << chunk_size << ", Time: " << elapsed_time << " seconds\n";

    for (int i = 0; i < omp_get_max_threads(); i++) {
        std::cout << "Thread " << i << ": " << total_iterations[i] << " iterations\n";
    }
    //saveDataToCSV(filename, schedule_type, chunk_size, num_threads,elapsed_time, total_iterations[i]);
    saveDataToCSV(filename, schedule_type, chunk_size, num_threads, elapsed_time, total_iterations, omp_get_max_threads());

    
    std::ofstream outfile("fractal_" + schedule_type + "_t" + std::to_string(num_threads) + "_ch" + std::to_string(chunk_size) + ".ppm", std::ios::out | std::ios::binary);
    outfile << "P6\n" << width << " " << height << "\n" << MaxColorComponentValue << "\n";
    outfile.write(reinterpret_cast<char*>(pixel_vec.data()), pixel_vec.size());
    outfile.close();
}

int main() {
    std::ofstream file(filename);

    std::vector<int> thread_counts = {1, 2, 4, 8}; 
    // std::vector<std::pair<std::string, int>> schedules = {
    //     {"static", 0}, {"static", 10}, {"dynamic", 0}, {"dynamic", 10}, {"guided", 0}, {"guided", 10}
    // };

    std::vector<std::string> schedules = {
        "static", "dynamic", "guided"
    };
    std::vector<int> chunks = {0,1,2,4,16,32,400,800};

    for (const auto& schedule_type : schedules) {
        for(const auto& chunk_size : chunks){
        //std::string schedule_type = schedule.first;
        //int chunk_size = schedule.second;

        
        if (schedule_type == "static") {
            omp_set_schedule(omp_sched_static, chunk_size);
        } else if (schedule_type == "dynamic") {
            omp_set_schedule(omp_sched_dynamic, chunk_size);
        } else if (schedule_type == "guided") {
            omp_set_schedule(omp_sched_guided, chunk_size);
        }       

        for (int threads : thread_counts) {
            computeMandelbrot(iXmax, iYmax, threads, schedule_type, chunk_size);
        }
        }
    }

    return 0;
}
