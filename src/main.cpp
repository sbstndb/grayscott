#include <iostream>
#include <omp.h>
#include <cmath>
#include <omp.h>
#include <vector>
#include <fstream>

const int N = 100;
const int STEPS = 10000;
const double Du = 0.16;
const double Dv = 0.08;
const double F = 0.035;
const double k = 0.065;

auto initialize(std::vector<std::vector<double>>& u, std::vector<std::vector<double>>& v) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i > N/2-10 && i < N/2+10 && j > N/2-10 && j < N/2+10) {
                u[i][j] = 0.5;
                v[i][j] = 0.25;
            } else {
                u[i][j] = 1.0;
                v[i][j] = 0.0;
            }
        }
    }
}

auto step(std::vector<std::vector<double>>& u, std::vector<std::vector<double>>& v) {
    std::vector<std::vector<double>> u_next = u;
    std::vector<std::vector<double>> v_next = v;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            double laplacian_u = u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - 4 * u[i][j];
            double laplacian_v = v[i-1][j] + v[i+1][j] + v[i][j-1] + v[i][j+1] - 4 * v[i][j];

            double u_val = u[i][j];
            double v_val = v[i][j];

            u_next[i][j] = u_val + (Du * laplacian_u - u_val * v_val * v_val + F * (1 - u_val));
            v_next[i][j] = v_val + (Dv * laplacian_v + u_val * v_val * v_val - (F + k) * v_val);
        }
    }

    u = u_next;
    v = v_next;
}


void save_grid(const std::vector<std::vector<double>>& grid, int step) {
    std::ofstream file("output_step_" + std::to_string(step) + ".txt");
    if (file.is_open()) {
        for (const auto& row : grid) {
            for (const auto& val : row) {
                file << val << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file for writing!" << std::endl;
    }
}



int main() {
    std::vector<std::vector<double>> u(N, std::vector<double>(N, 1.0));
    std::vector<std::vector<double>> v(N, std::vector<double>(N, 0.0));

    initialize(u, v);

    for (int step_num = 0; step_num < STEPS; ++step_num) {
        step(u, v);
        if (step_num % 1000 == 0) {
            std::cout << "Step: " << step_num << std::endl;
            save_grid(u, step_num);
        }
    }

    return 0;
}
