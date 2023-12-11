#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <functional>
#include <ctime>
#include <sys/time.h>
#include "omp.h"

using namespace std;

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}

// Function to initialize the 2D grid with an arbitrary function of i and j
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

// Function to print the 2D grid values
void printGrid(const vector<vector<double>>& grid, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            // cout << grid[i][j] << " ";
        }
        // cout << endl;
    }
}

// Function to calculate the residual of the 2D grid
void calculateResidual(const vector<vector<double>>& grid, const vector<vector<double>>& rhs, vector<vector<double>>& residual, int size) {
#pragma omp parallel for default(none) shared(residual, grid, rhs, size) collapse(2)
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            residual[i][j] = rhs[i][j] - (
                    grid[(i - 1 + size) % size][j] + // Left
                    grid[(i + 1) % size][j] +         // Right
                    grid[i][(j - 1 + size) % size] + // Up
                    grid[i][(j + 1) % size] -         // Down
                    4 * grid[i][j]
            );
        }
    }
}

bool check_stall(int x, int y, int size) {
    if (x == size / 4 && y == size / 4)
    {
        return true;
    }
    return false;
}

// Function to perform a relaxation step (Jacobi method) on the 2D grid with periodic boundary conditions
void relax(vector<vector<double>>& grid, const vector<vector<double>>& rhs, int size, int iterations, double& t, double tolerance) {
    double tmp = get_wall_time();
    vector<vector<double>> tempGrid = grid;

    bool exit_flag = false;
//#pragma omp parallel default(none) shared(iterations, size, tempGrid, grid, rhs, exit_flag, tolerance, cout)
            for (int iter = 0; iter < iterations; ++iter)
            {
#pragma omp parallel for default(none) shared(size, tempGrid, grid, rhs, exit_flag, tolerance)
                for (int i = 0; i < size; ++i)
                {
                    if(exit_flag)
                    {
                        continue;
                    }
                    for (int j = 0; j < size; ++j)
                    {
                        tempGrid[i][j] = 0.25 * (
                                grid[(i - 1 + size) % size][j] + // Left
                                grid[(i + 1) % size][j] +         // Right
                                grid[i][(j - 1 + size) % size] + // Up
                                grid[i][(j + 1) % size] -         // Down
                                rhs[i][j]
                        );
                        if(check_stall(i, j, size) && abs(grid[i][j] - tempGrid[i][j]) < tolerance)
                        {
                            exit_flag = true;
                            break;
                        }
                    }
                }
                if(exit_flag)
                {
                    cout << "break at iter: " << iter << endl;
                    break;
                }
                grid = tempGrid;
            }

    t += get_wall_time() - tmp;
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

#pragma omp parallel for default(none) shared(coarseSizeX, coarseSizeY, fineGrid, coarseGrid, fineSizeX, fineSizeY) collapse(2)
    for (int i = 0; i < fineSizeX; ++i)
    {
        for (int j = 0; j < fineSizeY; ++j)
        {
            fineGrid[i][j] = 0.25 * (
                    coarseGrid[i / 2][j / 2] +             // Center
                    coarseGrid[(i / 2 + 1) % coarseSizeX][j / 2] +             // Right
                    coarseGrid[i / 2][(j / 2 + 1) % coarseSizeY] +             // Down
                    coarseGrid[(i / 2 + 1) % coarseSizeX][(j / 2 + 1) % coarseSizeY]  // Bottom-right
            );
        }
    }
}

// Recursive function for the V-cycle multigrid solver for 2D Poisson equation
void vCycle(vector<vector<double>>& grid, const vector<vector<double>>& rhs, int size, int maxIterations, int currentLevel, int maxLevels, double& t, double tolerance) {
    if (currentLevel == maxLevels) {
        // At the coarsest level, perform relaxation directly
        relax(grid, rhs, size, maxIterations, t, tolerance);
    } else {
        // Pre-smooth with relaxation
        relax(grid, rhs, size, maxIterations, t, tolerance);

        // Calculate the residual
        vector<vector<double>> residual(size, vector<double>(size, 0.0));
        calculateResidual(grid, rhs, residual, size);

        // Restrict the residual to the next coarser level
        int coarseSizeX = (size - 1) / 2 + 1;
        int coarseSizeY = (size - 1) / 2 + 1;
        vector<vector<double>> coarseResidual(coarseSizeX, vector<double>(coarseSizeY, 0.0));
        restrictResidual(residual, coarseResidual);

        // Recursive call to the next level
        vector<vector<double>> coarseGrid(coarseSizeX, vector<double>(coarseSizeY, 0.0));
        vCycle(coarseGrid, coarseResidual, coarseSizeX, maxIterations, currentLevel + 1, maxLevels, t, tolerance);

        // Interpolate the correction and add it to the current grid
        vector<vector<double>> correction(size, vector<double>(size, 0.0));
        interpolateGrid(coarseGrid, correction);

#pragma omp parallel for default(none) shared(grid, correction, size) collapse(2)
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                grid[i][j] += correction[i][j];
            }
        }

        // Post-smooth with relaxation
        relax(grid, rhs, size, maxIterations, t, tolerance);
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
        cout << "Solution written to " << filename << endl;
    } else {
        cout << "Unable to open the file.." << endl;
    }
}

// Example of an arbitrary initialization function (you can define your own)
inline double arbitraryInitialization(int i, int j) {
    // For demonstration purposes, let's initialize with the sum of i and j
    return static_cast<double>(0*(i + j)); // replace (i+j) with any f(i,j)
}

// Example of an arbitrary rhs initialization function (you can define your own)
double arbitraryRHS(int i, int j, int size) {
    // For demonstration purposes, let's initialize with a function of i and j
    return static_cast<double>(sin(i*2*3.1415/(size))*sin(j*2*3.1415/(size)));
}

template <typename T>
void retrieve_arg(const string& arg, T& prec) {
    try {
        size_t pos;
        if (std::is_same_v<T, double>)
        {
            prec = stod(arg, &pos);
        }
        if (is_same_v<T, int>)
        {
            prec = stoi(arg, &pos);
        }
        if (pos < arg.size()) {
            cerr << "Trailing characters after number: " << arg << '\n';
        }
    } catch (invalid_argument const &ex) {
        cerr << "Invalid number: " << arg << '\n';
    } catch (out_of_range const &ex) {
        cerr << "Number out of range: " << arg << '\n';
    }
}

int main(int argc, char** argv) {
    double tmp_time = get_wall_time();

    // Size of the 2D grid (including boundary points)
    int size;
    retrieve_arg(argv[1], size);
    cout << "problem size: " << size << endl;

    // define the num of threads for openmp
    int num_thread;
    retrieve_arg(argv[2], num_thread);
    cout << "num of threads: " << num_thread << endl;
    omp_set_num_threads(num_thread);

    double tolerance;
    retrieve_arg(argv[3], tolerance);
    cout << "tolerance: " << tolerance << endl;

    const int maxIterations = 3000;  // Number of relaxation iterations
    const int maxLevels = 3;  // Number of multigrid levels
    const double pi = 3.1415926;
    double time_relax = 0.0;

    // Initialize the 2D grid with an arbitrary function
    vector<vector<double>> grid;

    initializeGrid(grid, size, arbitraryInitialization);
    double seq_time = get_wall_time() - tmp_time;
    cout << "leave sequential init: " <<  seq_time << endl;


    tmp_time = get_wall_time();
    // Initialize the right-hand side source with an arbitrary function
    vector<vector<double>> rhs(size, vector<double>(size, 0.0));

#pragma omp parallel for default(none) shared(size, rhs, pi)
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            rhs[i][j] = sin(i * 2 * pi / (size)) * \
            sin(j * 2 * pi / (size));
        }
    }


    // cout << "enter iteration: " << get_wall_time();

    // Perform V-cycle multigrid solver with right-hand side source
    vCycle(grid, rhs, size, maxIterations, 1, maxLevels, time_relax, tolerance);

    double par_time = get_wall_time() - tmp_time;
    cout << "leave parallel part: " << par_time << endl;

    cout << "jacobi time: " << time_relax << endl;

    double P = par_time / (par_time + seq_time);
    cout << "overall speedup: "  << 1 / (1 - P + P / num_thread) << endl;
    // cout << "parallel efficiency: " << speedup / num_thread << endl;
    // Print the final solution
    //cout << "Final Solution:" << endl;
    //printGrid(grid, size);

    // Write the solution to a text file
    writeToFile(grid, size, "solution.txt");

    return 0;
}
