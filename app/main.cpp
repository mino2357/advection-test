#include <iostream>
#include <array>
#include <cmath>
#include <limits>
#include <iomanip>

const double Lx = 1.0;
const double Ly = 1.0;
const int Nx    = 64;
const int Ny    = 64;
const double dx = Lx / (Nx - 1);
const double dy = Ly / (Ny - 1);
const double dt = 0.0001;
const int INTV  = 100;

namespace rittai3d{
	namespace utility{
		// [ minimum, maximum ) の範囲でラップアラウンド
		template <typename T>
		constexpr T wrap_around(T value, T minimum, T maximum){
			const T n = (value - minimum) % (maximum - minimum);
			return n >= 0 ? (n + minimum) : (n + maximum); 
		}
	}
}

namespace sksat {
	template<std::size_t Num, typename T = double>
	class array_wrapper{
	private:
		std::array<T, Num> arr;
	public:
		constexpr array_wrapper() : arr() {}
		~array_wrapper() = default;

		T& operator[](int i){
			i = rittai3d::utility::wrap_around(i, 0, static_cast<int>(Num));
			return arr[i];
		}
	};
}

using Array = sksat::array_wrapper<Nx, sksat::array_wrapper<Ny>>;;

void upWind_firstOrder_X(Array& u, Array& v, Array& u_next){
    double ad1, ad2;
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            ad1 = 1.0; // u[i][j];
            ad2 = 1.0; // v[i][j];
            u_next[i][j] = u[i][j]
                            - dt * ( ad1 * (u[i+1][j] - u[i-1][j]) / (2.0 * dx) + std::abs(ad1) * (- u[i+1][j] + 2.0 * u[i][j] - u[i-1][j]) / (2.0 * dx))
                            - dt * ( ad2 * (u[i][j+1] - u[i][j-1]) / (2.0 * dy) + std::abs(ad2) * (- u[i][j+1] + 2.0 * u[i][j] - u[i][j-1]) / (2.0 * dy));
        }
    }
}

void upWind_firstOrder_Y(Array& u, Array& v, Array& v_next){
    double ad1, ad2;
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            ad1 = 1.0; // u[i][j];
            ad2 = 1.0; // v[i][j];
            v_next[i][j] = v[i][j]
                            - dt * ( ad1 * (v[i+1][j] - v[i-1][j]) / (2.0 * dx) + std::abs(ad1) * (- v[i+1][j] + 2.0 * v[i][j] - v[i-1][j]) / (2.0 * dx))
                            - dt * ( ad2 * (v[i][j+1] - v[i][j-1]) / (2.0 * dy) + std::abs(ad2) * (- v[i][j+1] + 2.0 * v[i][j] - v[i][j-1]) / (2.0 * dy));
        }
    }
}

void clear(Array& u, Array& v, Array& u_next, Array& v_next){
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u[i][j] = u_next[i][j];
            v[i][j] = v_next[i][j];
        }
    }
}

void init_func(Array& u, Array& v){
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            if(i>Nx/4 && i<3*Nx/4 && j>Ny/4 && j<3*Ny/4){
                u[i][j] = 1.0;
                v[i][j] = 1.0;
            }
        }
    }
}

int main(){
    std::cout << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 - 8);

    Array u = {};
    Array v = {};
    Array u_next = {};
    Array v_next = {};

    init_func(u, v);

    /**********************************************************************/
    /*                 可視化の設定(gnuplot)                                */
    /**********************************************************************/
    std::FILE *gp = popen( "gnuplot -persist", "w" );
    fprintf(gp, "set xr [0:%f]\n", Lx);
    fprintf(gp, "set yr [0:%f]\n", Ly);
    fprintf(gp, "set contour\n");
    fprintf(gp, "set grid\n");
    fprintf(gp, "unset key\n");
    fprintf(gp, "set size ratio 1\n");
    fprintf(gp, "set palette rgb 33,13,10\n");
    
    //初期条件描画
    double norm = 0.0;
    double coef = 0.0;
    fprintf(gp, "plot '-' with vector lc palette\n");
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            norm = std::sqrt(u[i][j] * u[i][j] + v[i][j] * v[i][j]);
            fprintf(gp, "%f %f %f %f %f\n", i * dx , j * dy, u[i][j] / (norm * Nx), v[i][j] / (norm * Ny), norm);
        }
    }
    fprintf(gp, "e\n");
    fflush(gp);

    //std::cout << "Enterキーを押してください．" << std::endl;
    //getchar();

    for(int i=0;;++i){
        upWind_firstOrder_X(u ,v, u_next);
        upWind_firstOrder_Y(u ,v, v_next);
        clear(u, v, u_next, v_next);

        // 描画
        if(i%INTV == 0){
            fprintf(gp, "plot '-' with vector lc palette\n");
            for(int i=0; i<Nx; ++i){
                for(int j=0; j<Ny; ++j){
                    norm = std::sqrt(u[i][j] * u[i][j] + v[i][j] * v[i][j]);
                    fprintf(gp, "%f %f %f %f %f\n", i * dx , j * dy, u[i][j] / (norm * Nx), v[i][j] / (norm * Ny), norm);
                }
            }
            fprintf(gp, "e\n");
            fflush(gp);
        }
    }
}