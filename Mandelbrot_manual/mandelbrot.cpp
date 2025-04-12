
//PROGRAM GENERATING MANDELBROT SET IN PARALLEL, WITH MANUAL THREAD CREATION.
//RESULTING PPM IS COLOURED ACCORDING TO THREAD TID

//-lpthread $(pkg-config --cflags --libs opencv4)



#include <stdio.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include <chrono>  


const int iXmax = 800;
const int iYmax = 800;
const int MaxColorComponentValue = 255;
const int IterationMax = 200;
const double EscapeRadius = 2;
double ER2 = EscapeRadius * EscapeRadius;
// std::mutex mutex;

unsigned char thread_colors[12][3] = {
    {255, 0, 0},    
    {0, 255, 0},    
    {0, 0, 255},    
    {255, 255, 0},  
    {255, 0, 255},  
    {0, 255, 255},  
    {128, 0, 0},    
    {0, 128, 0},   
    {0, 0, 128},    
    {128, 128, 0},  
    {128, 0, 128},  
    {0, 128, 128}   
};

void generateFractalSection(int thread_id, int startY, int endY, std::vector<unsigned char>& pixels) {
    auto start_time = std::chrono::high_resolution_clock::now(); 

    double Cx, Cy;
    double PixelWidth = (1.5 - (-2.5)) / iXmax;
    double PixelHeight = (2.0 - (-2.0)) / iYmax;
    double Zx, Zy, Zx2, Zy2;
    int Iteration;

    unsigned char color[3] = {thread_colors[thread_id][0], thread_colors[thread_id][1], thread_colors[thread_id][2]};
    
    int index = startY * iXmax * 3;  // Starting index
    for (int iY = startY; iY < endY; ++iY) {
        Cy = -2.0 + iY * PixelHeight;
        if (fabs(Cy) < PixelHeight / 2) Cy = 0.0; // Main antenna
        
        for (int iX = 0; iX < iXmax; ++iX) {
            Cx = -2.5 + iX * PixelWidth;
            Zx = 0.0;
            Zy = 0.0;
            Zx2 = Zx * Zx;
            Zy2 = Zy * Zy;
            
            for (Iteration = 0; Iteration < IterationMax && ((Zx2 + Zy2) < ER2); Iteration++) {
                Zy = 2 * Zx * Zy + Cy;
                Zx = Zx2 - Zy2 + Cx;
                Zx2 = Zx * Zx;
                Zy2 = Zy * Zy;
            }

            if (Iteration == IterationMax) {
                // Mandelbrot set = black
                pixels[index] = 0;
                pixels[index + 1] = 0;
                pixels[index + 2] = 0;
            } else {
                // Outside Mandelbrot set = color
                pixels[index] = color[0];
                pixels[index + 1] = color[1];
                pixels[index + 2] = color[2];
            }
            // background[index] = color[0];
            // background[index + 1] = color[1];
            // background[index + 2] = color[2];
            index += 3;  
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> duration = end_time - start_time; 
    std::cout << "Thread " << thread_id << " - " << duration.count() << " s.\n"; 
}

int main() {
    std::ofstream outfile("fractal.ppm", std::ios::out | std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "ERROR - failed to open the file.\n";
        return 1;
    }

    outfile << "P6\n" << iXmax << " " << iYmax << "\n" << MaxColorComponentValue << "\n";

    std::vector<unsigned char> pixel_vec(3 * iXmax * iYmax);
    // std::vector<unsigned char> background_vec(3 * iXmax * iYmax);

    std::vector<std::thread> threads;
    int num_threads = 12;
    int rows_per_thread = iYmax / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int startY = i * rows_per_thread;
        int endY = (i == num_threads - 1) ? iYmax : startY + rows_per_thread;
        threads.push_back(std::thread(generateFractalSection, i, startY, endY, std::ref(pixel_vec)));
    }

    for (auto& t : threads) {
        t.join();
    }

    // Make image buffer
    cv::Mat image(iYmax, iXmax, CV_8UC3, pixel_vec.data());

    if (!cv::imwrite("mandelbrot.png", image)) {
        std::cerr << "Error saving the png image.\n";
    }

    outfile.write(reinterpret_cast<char*>(pixel_vec.data()), pixel_vec.size());
    outfile.close();

    return 0;
}
