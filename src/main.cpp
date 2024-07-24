#include <iostream>
#include <omp.h>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std ; 


template <typename FLOAT>
class GS {
public:
	int N = 4096;
	int STEPS = 1000;
	FLOAT Du = 0.16;
	FLOAT Dv = 0.08;
	FLOAT F = 0.035;
	FLOAT k = 0.065;

	vector<vector<FLOAT>>u, v, u_next, v_next;


auto initialize() {
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

auto step() {
    u_next = u;
    v_next = v;
    #pragma omp parallel for schedule(static) shared(u, v, u_next, v_next)
    for (int i = 1; i < N - 1; ++i) {
        #pragma omp simd
        for (int j = 1; j < N - 1; ++j) {
            FLOAT laplacian_u = u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - 4 * u[i][j];
            FLOAT laplacian_v = v[i-1][j] + v[i+1][j] + v[i][j-1] + v[i][j+1] - 4 * v[i][j];

            FLOAT u_val = u[i][j];
            FLOAT v_val = v[i][j];

            u_next[i][j] = u_val + (Du * laplacian_u - u_val * v_val * v_val + F * (1 - u_val));
            v_next[i][j] = v_val + (Dv * laplacian_v + u_val * v_val * v_val - (F + k) * v_val);
        }
    }

    u = u_next;
    v = v_next;
}


auto step_blocking(int BLOCK_SIZE){
//    int BLOCK_SIZE = 32;

    u_next = u ; 
    v_next = v ; 

    #pragma omp for schedule(static)
    for (int ii = 1 ; ii < N-1 ; ii += BLOCK_SIZE){
        for (int jj = 1 ; jj < N-1 ; jj += BLOCK_SIZE){
            for (int i = ii; i < min(ii+BLOCK_SIZE, N-1); i+=1){
		#pragma omp simd
                for (int j = jj; j < min(jj+BLOCK_SIZE, N-1); j+=1){
                    double laplacian_u = u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - 4 * u[i][j];
                    double laplacian_v = v[i-1][j] + v[i+1][j] + v[i][j-1] + v[i][j+1] - 4 * v[i][j];
                    double u_val = u[i][j];
                    double v_val = v[i][j];
                    u_next[i][j] = u_val + (Du * laplacian_u - u_val * v_val * v_val + F * (1 - u_val));
                    v_next[i][j] = v_val + (Dv * laplacian_v + u_val * v_val * v_val - (F + k) * v_val);
                }
            }
        }
    }
    u = u_next;
    v = v_next;
}


void save_grid(const std::vector<std::vector<FLOAT>>& grid, int step) {
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

private:

};



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

    using FLOAT = float ; 

    GS<FLOAT> gs ;
    gs.u = vector<vector<FLOAT>>(gs.N, vector<FLOAT>(gs.N, 1.0));
    gs.v = vector<vector<FLOAT>>(gs.N, vector<FLOAT>(gs.N, 0.0));

    gs.initialize();
    #pragma omp parallel
    for (int step_num = 0; step_num < gs.STEPS; ++step_num) {
        gs.step();
	//gs.step_blocking(32);
        if (step_num % 1000 == 0) {
            std::cout << "Step: " << step_num << std::endl;
//            gs.save_grid(gs.u, step_num);
        }
    }

    return 0;
}
