#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <functional>
#include <time.h>
#include <sys/time.h>
#include "omp.h"

using namespace std;

const double pi = 3.1415926;

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}

// Function to calculate the standard deviation of differences between values in the solution matrix and an arbitrary function
double calculateStandardDeviation(const vector<vector<double>>& solution, function<double(int, int, int)> AnalSol) {
    int size = solution.size();

    // Calculate the differences and store them in a vector
    vector<double> differences;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double diff = solution[i][j] - AnalSol(i, j, size);
            differences.push_back(diff);
        }
    }

    // Calculate the mean of the differences
    double mean = 0.0;
    for (double diff : differences) {
        mean += diff;
    }
    mean /= (size * size);

    // Calculate the sum of squared differences from the mean
    double sumSquaredDifferences = 0.0;
    for (double diff : differences) {
        double squaredDifference = (diff - mean) * (diff - mean);
        sumSquaredDifferences += squaredDifference;
    }

    // Calculate the variance and then the standard deviation
    double variance = sumSquaredDifferences / (size * size);
    double standardDeviation = sqrt(variance)/0.632007 * 100;

    return standardDeviation;
}

// Exact solution function, used for error computation
double AnalSol(int i, int j, int size) {
    return static_cast<double>(-100*sin(2*pi*((i)%size)/size)*sin(2*pi*((j) % size)/size)/(8*pi*pi));

}

// Function to initialize the 2D grid with a function of i and j
void initializeGrid(vector<vector<double>>& grid, int size, function<double(int, int)> initFunction) {
    for (int i = 0; i < size; ++i)
    {
        grid.push_back(vector<double>(size, 0.0));
        for (int j = 0; j < size; ++j)
        {
            grid[i][j] = initFunction(i, j);
        }
    }
}

// Function to calculate the residual of the 2D grid
void calculateResidual(const vector<vector<double>>& grid, const vector<vector<double>>& rhs, vector<vector<double>>& residual, int size) {
#pragma omp parallel for default(none) shared(residual, grid, rhs, size) collapse(2)
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            residual[i][j] = rhs[i][j] - size * size * (
                    grid[(i - 1 + size) % size][j] + // Left
                    grid[(i + 1) % size][j] +         // Right
                    grid[i][(j - 1 + size) % size] + // Up
                    grid[i][(j + 1) % size]	// Down
                    - 4 * grid[i][j]
            );
        }
    }
}

void relax(vector<vector<double>>& grid, const vector<vector<double>>& rhs, int size, int iterations, double& t, int l, int k) {
    vector<vector<double>> tempGrid = grid;


    for (int iter = 0; iter < iterations; ++iter) {
#pragma omp parallel for default(none) shared(size, tempGrid, grid, rhs) collapse(2)
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                tempGrid[i][j] = (2.0 / 3) * 0.25 * (
                        grid[(i - 1 + size) % size][j] + // Left
                        grid[(i + 1) % size][j] +         // Right
                        grid[i][(j - 1 + size) % size] + // Up
                        grid[i][(j + 1) % size] -         // Down
                        (1.0 / (size * size)) * rhs[i][j]
                ) + (1.0 / 3) * grid[i][j];
            }
        }
        grid = tempGrid;
    }
}



// Function to restrict the 2D grid to a coarser level using the residual
void restrictResidual(const vector<vector<double>>& residual, vector<vector<double>>& coarseResidual) {
    int coarseSizeX = coarseResidual.size();
    int coarseSizeY = coarseResidual[0].size();

#pragma omp parallel for default(none) shared(coarseSizeX, coarseSizeY, residual, coarseResidual) collapse(2)
    for (int i = 0; i < coarseSizeX; ++i)
    {
        for (int j = 0; j < coarseSizeY; ++j)
        {
            // keep injection, but create a temp grid to create the resitriction properly first and then inject
            coarseResidual[i][j] = residual[2 * i][2 * j];
        }
    }
}

// Function to interpolate the 2D grid to a finer level
void interpolateGrid(const vector<vector<double>>& coarseGrid, vector<vector<double>>& fineGrid) {
    int coarseSizeX = coarseGrid.size();
    int coarseSizeY = coarseGrid[0].size();
    int fineSizeX = fineGrid.size();
    int fineSizeY = fineGrid[0].size();

#pragma omp parallel for default(none) shared(coarseSizeX, coarseSizeY, fineGrid, coarseGrid, fineSizeX, fineSizeY)
    for (int i = 0; i < fineSizeX; ++i) {
        for (int j = 0; j < fineSizeY; ++j) {
            fineGrid[i][j] = 0.25 * (
                    coarseGrid[i / 2][j / 2] +                         // UpLeft
                    coarseGrid[(i / 2 + 1) % coarseSizeX][j / 2] +     // BottomLeft
                    coarseGrid[i / 2][(j / 2 + 1) % coarseSizeY] +     // UpRight
                    coarseGrid[(i / 2 + 1) % coarseSizeX][(j / 2 + 1) % coarseSizeY]    // BottomRight

                    // coarseGrid[(i / 2 - 1 + coarseSizeX) % coarseSizeX][j / 2] +  // Left WRONG
                    // coarseGrid[i / 2][(j / 2 - 1 + coarseSizeY) % coarseSizeY] +  // Up WRONG
            );
        }
    }
}



// Recursive function for the V-cycle multigrid solver for 2D Poisson equation

void vCycle(vector<vector<double>>& grid, const vector<vector<double>>& rhs, int size, int maxIterations, int currentLevel, int maxLevels, double& t) {
    if (currentLevel == maxLevels) {
        // At the coarsest level, perform relaxation directly
        relax(grid, rhs, size, maxIterations, t, currentLevel, maxLevels);
    }   else{
        // Pre-smooth with relaxation
        relax(grid, rhs, size, maxIterations, t, currentLevel, maxLevels);

        // Calculate the residual
        vector<vector<double>> residual(size, vector<double>(size, 0.0));
        calculateResidual(grid, rhs, residual, size);

        // Restrict the residual to the next coarser level
        int coarseSizeX = (size) / 2 ;
        int coarseSizeY = (size) / 2 ;
        vector<vector<double>> coarseResidual(coarseSizeX, vector<double>(coarseSizeY, 0.0));
        restrictResidual(residual, coarseResidual);

        // Recursive call to the next level
        vector<vector<double>> coarseGrid(coarseSizeX, vector<double>(coarseSizeY, 0.0));
        vCycle(coarseGrid, coarseResidual, coarseSizeX, maxIterations, currentLevel + 1, maxLevels, t);

        // Interpolate the correction and add it to the current grid
        vector<vector<double>> correction(size, vector<double>(size, 0.0));
        interpolateGrid(coarseGrid, correction);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                grid[i][j] += correction[i][j];
            }
        };

        // Post-smooth with relaxation
        relax(grid, rhs, size, maxIterations, t, currentLevel, maxLevels);
    }
}


// Function to write the 2D grid to a text file
void writeToFile(const vector<vector<double>>& grid, int size, const string& filename) {

    ofstream outFile(filename);

    if (outFile.is_open())
    {
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                outFile << grid[i][j] << " ";
            }
            outFile << endl;
        }
        outFile.close();
    };
}

// Example of an initialization function
double InitialGuess(int i, int j) {
    return static_cast<double>(0*(i + j)); // replace (i+j) with any f(i,j), currently it is just zero
}

// RIGHT HAND SIDE
double RHS(int i, int j, int size) {
    return static_cast<double>(100*sin(((i)%size)*2*pi/(size))*sin(((j)%size)*2*pi/(size))); //SOURCE HAS TO BE PERIODIC, ELSE NO CONVERGENCE.
}

int main() {
    int size;  // Size of the 2D grid (including boundary points)
    int maxIterations;  // Number of relaxation iterations
    int maxLevels;  // Number of multigrid levels
    double time_relax = 0.0;
    int n = 0; // number of v-cycles done
    ofstream ERROR("error.txt"), LOGERROR("logerror.txt");


    cout << endl;
    cout << " + Side size: ";
    cin >> size; size = pow(2,int(log2(size)));
    cout << " + Enter the number of threads: ";
    int numThreads;
    cin >> numThreads;

    // Set the number of threads
    omp_set_num_threads(numThreads);

    // cout << " + No. of smoother iterations per level: ";
    maxIterations = 5; // cin >> maxIterations;

    // cout << " + No. of grids (Max is " << log2(size) << "): ";
    // cin >> maxLevels;

    // if(maxLevels > log2(size)){maxLevels = log2(size);}; // if too many levels requested, return to max possible.
    maxLevels = log2(size);

    // Initialize the 2D grid
    vector<vector<double>> grid;
    double tmp_time = get_wall_time();
    initializeGrid(grid, size, InitialGuess);

    // Initialize the right-hand side source with an arbitrary function
    vector<vector<double>> rhs(size, vector<double>(size, 0.0));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            rhs[i][j] = RHS(i, j, size);
        }
    }

    // Perform V-cycle multigrid solver

    double stdDev = calculateStandardDeviation(grid, AnalSol);
    double stdDev0 = stdDev + 1; // just make sure they are properly different at the start

    cout << endl << " ~ The solver is initialized to: [" << size << " x " << size << " grid (" << size*size << " pts), " << maxLevels << " levels, " << numThreads << " threads]" << endl;

    while(stdDev < stdDev0) // Iterate while error normal is decreasing
    {
        stdDev0 = stdDev;
        n++;
        // ERROR << stdDev << endl; OUTPUT ERROR INFORMATION
        // LOGERROR << log((stdDev)) << endl; LOG OF THE ABOVE
        vCycle(grid, rhs, size, maxIterations, 1, maxLevels, stdDev);
        stdDev = calculateStandardDeviation(grid, AnalSol);
    };


    if(stdDev > 110){ cout << endl << " !!! DIVERGENT BEHAVIOUR. CHANGE IN PARAMETERS IS REQUIRED !!! " << endl; };

    cout << endl;
    cout << " - Time taken: " << get_wall_time() - tmp_time << " seconds." << endl;
    cout << " - Number of v-cycles done: " << n << " v-cycles." << endl;
    // cout << " - Number of smoother iterations done: " <<  n * maxIterations * 2*maxLevels - 1  << " sweeps." << endl;

    // write the solution to a text file
    // writeToFile(grid, size, "solution.txt");

    // Output the accuracy measure
    // stdDev = calculateStandardDeviation(grid, AnalSol);

    cout << " - Final Error Estimate: ~" << 100*pow(0.5,n) << "%" << endl << endl << endl;

    return 0;
}