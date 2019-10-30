/*
 * 移流のテスト
 * 
 * 移流方程式/バーガース方程式
 */

#include <iostream>
#include <array>
#include <cmath>
#include <limits>
#include <iomanip>

// parameters
const double Lx    = 1.0;
const double Ly    = 1.0;
const int Nx       = 64;
const int Ny       = 64;
const double dx    = Lx / (Nx - 1);
const double dy    = Ly / (Ny - 1);
const double a     = 1.0; // advection [a, b]^T
const double b     = 1.0; // vector field
const double alpha = 1.0;   // 1:UTOPIA 3:K-K
const double dt    = 0.01 / (std::sqrt(a * a + b * b) * std::sqrt((1.0 / dx) * (1.0 / dx) + (1.0 / dy) * (1.0 / dy)));
const int INTV     = 100;

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

/********************************/
/*      up wind scheme          */
/********************************/

void upWind_firstOrder_X(Array& u, Array& v, Array& u_next){
    double ad1, ad2;
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            ad1 = a; // u[i][j];
            ad2 = b; // v[i][j];
            u_next[i][j] =
                - ( ad1 * (u[i+1][j] - u[i-1][j]) / (2.0 * dx) + std::abs(ad1) * (- u[i+1][j] + 2.0 * u[i][j] - u[i-1][j]) / (2.0 * dx))
                - ( ad2 * (u[i][j+1] - u[i][j-1]) / (2.0 * dy) + std::abs(ad2) * (- u[i][j+1] + 2.0 * u[i][j] - u[i][j-1]) / (2.0 * dy));
        }
    }
}

void upWind_firstOrder_Y(Array& u, Array& v, Array& v_next){
    double ad1, ad2;
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            ad1 = a; // u[i][j];
            ad2 = b; // v[i][j];
            v_next[i][j] =
                - ( ad1 * (v[i+1][j] - v[i-1][j]) / (2.0 * dx) + std::abs(ad1) * (- v[i+1][j] + 2.0 * v[i][j] - v[i-1][j]) / (2.0 * dx))
                - ( ad2 * (v[i][j+1] - v[i][j-1]) / (2.0 * dy) + std::abs(ad2) * (- v[i][j+1] + 2.0 * v[i][j] - v[i][j-1]) / (2.0 * dy));
        }
    }
}

/********************************/
/*      K-K/UTOPIA scheme       */
/********************************/

void upWind_thirdOrder_X(Array& u, Array& v, Array& u_next){
    double ad1, ad2;
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            ad1 = a; // u[i][j];
            ad2 = b; // v[i][j];
            u_next[i][j] =
                - ( ad1 * ( - u[i+2][j] + 8.0 * (u[i+1][j] - u[i-1][j]) + u[i-2][j]) / (12.0 * dx)
                    + alpha * std::abs(ad1) * ( u[i+2][j] - 4.0 * u[i+1][j] + 6.0 * u[i][j] - 4.0 * u[i-1][j] + u[i-2][j]) / (12.0 * dx))
                - ( ad2 * ( - u[i][j+2] + 8.0 * (u[i][j+1] - u[i][j-1]) + u[i][j-2]) / (12.0 * dy)
                    + alpha * std::abs(ad2) * ( u[i][j+2] - 4.0 * u[i][j+1] + 6.0 * u[i][j] - 4.0 * u[i][j-1] + u[i][j-2]) / (12.0 * dy));
        }
    }
}

void upWind_thirdOrder_Y(Array& u, Array& v, Array& v_next){
    double ad1, ad2;
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            ad1 = a; // u[i][j];
            ad2 = b; // v[i][j];
            v_next[i][j] =
                - ( ad1 * (- v[i+2][j] + 8.0 * (v[i+1][j] - v[i-1][j]) + v[i-2][j] ) / (12.0 * dx)
                    + alpha * std::abs(ad1) * ( v[i+2][j] - 4.0 * v[i+1][j] + 6.0 * v[i][j] - 4.0 * v[i-1][j] + v[i-2][j]) / (12.0 * dx))
                - ( ad2 * ( - v[i][j+2] + 8.0 * (v[i][j+1] - v[i][j-1]) + v[i][j-2]) / (12.0 * dy)
                    + alpha * std::abs(ad2) * ( v[i][j+2] - 4.0 * v[i][j+1] + 6.0 * v[i][j] - 4.0 * v[i][j-1] + v[i][j-2]) / (12.0 * dy));
        }
    }
}

/********************************/
/*      up wind 5th scheme      */
/********************************/

void upWind_fifthOrder_X(Array& u, Array& v, Array& u_next){
    double ad1, ad2;
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            ad1 = a; // u[i][j];
            ad2 = b; // v[i][j];
            u_next[i][j] =
                - ( ad1 * ( u[i+3][j] - 9.0 * u[i+2][j] + 45.0 * (u[i+1][j] - u[i-1][j]) + 9.0 * u[i-2][j] - u[i-3][j] ) / (60.0 * dx)
                    + std::abs(ad1) * ( - u[i+3][j] + 6.0 * u[i+2][j] - 15.0 * u[i+1][j] + 20.0 * u[i][j] - 15.0 * u[i-1][j] + 6.0 * u[i-2][j] - u[i-3][j] ) / (60.0 * dx))
                - ( ad2 * ( u[i][j+3] - 9.0 * u[i][j+2] + 45.0 * (u[i][j+1] - u[i][j-1]) + 9.0 * u[i][j-2] - u[i][j-3] ) / (60.0 * dy)
                    + std::abs(ad2) * ( - u[i][j+3] + 6.0 * u[i][j+2] - 15.0 * u[i][j+1] + 20.0 * u[i][j] - 15.0 * u[i][j-1] + 6.0 * u[i][j-2] - u[i][j-3] ) / (60.0 * dy));
        }
    }
}

void upWind_fifthOrder_Y(Array& u, Array& v, Array& v_next){
    double ad1, ad2;
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            ad1 = a; // u[i][j];
            ad2 = b; // v[i][j];
            v_next[i][j] =
                - ( ad1 * ( v[i+3][j] - 9.0 * v[i+2][j] + 45.0 * (v[i+1][j] - v[i-1][j]) + 9.0 * v[i-2][j] - v[i-3][j] ) / (60.0 * dx)
                    + std::abs(ad1) * ( - v[i+3][j] + 6.0 * v[i+2][j] - 15.0 * v[i+1][j] + 20.0 * v[i][j] - 15.0 * v[i-1][j] + 6.0 * v[i-2][j] - v[i-3][j] ) / (60.0 * dx))
                - ( ad2 * ( v[i][j+3] - 9.0 * v[i][j+2] + 45.0 * (v[i][j+1] - v[i][j-1]) + 9.0 * v[i][j-2] - v[i][j-3] ) / (60.0 * dy)
                    + std::abs(ad2) * ( - v[i][j+3] + 6.0 * v[i][j+2] - 15.0 * v[i][j+1] + 20.0 * v[i][j] - 15.0 * v[i][j-1] + 6.0 * v[i][j-2] - v[i][j-3] ) / (60.0 * dy));
        }
    }
}

// clear
void clear(Array& u, Array& v, Array& u_next, Array& v_next){
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u[i][j] = u_next[i][j];
            v[i][j] = v_next[i][j];
        }
    }
}

// initialize function
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

// Schemes
void Euler_1stOrder(Array& u, Array& v, Array& u_next, Array& v_next){
    Array du = {};
    Array dv = {};
    upWind_firstOrder_X(u, v, du);
    upWind_firstOrder_Y(u, v, dv);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u_next[i][j] = u[i][j] + dt * du[i][j];
            v_next[i][j] = v[i][j] + dt * dv[i][j];
        }
    }
}

void Euler_3rdOrder(Array& u, Array& v, Array& u_next, Array& v_next){
    Array du = {};
    Array dv = {};
    upWind_thirdOrder_X(u, v, du);
    upWind_thirdOrder_Y(u, v, dv);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u_next[i][j] = u[i][j] + dt * du[i][j];
            v_next[i][j] = v[i][j] + dt * dv[i][j];
        }
    }
}

void TVD_RK3_3rdOrder(Array& u, Array& v, Array& u_next, Array& v_next){
    Array du1 = {};
    Array dv1 = {};
    Array u1  = {};
    Array v1  = {};
    upWind_thirdOrder_X(u ,v, du1);
    upWind_thirdOrder_Y(u ,v, dv1);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u1[i][j] = u[i][j] + dt * du1[i][j];
            v1[i][j] = v[i][j] + dt * dv1[i][j];
        }
    }
    Array du2 = {};
    Array dv2 = {};
    Array u2  = {};
    Array v2  = {};
    upWind_thirdOrder_X(u1 ,v1, du2);
    upWind_thirdOrder_Y(u1 ,v1, dv2);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u2[i][j] = 3.0 / 4.0 * u[i][j] + 1.0 / 4.0 * u1[i][j] + 1.0 / 4.0 * dt * du2[i][j];
            v2[i][j] = 3.0 / 4.0 * v[i][j] + 1.0 / 4.0 * v1[i][j] + 1.0 / 4.0 * dt * dv2[i][j];
        }
    }
    Array du3 = {};
    Array dv3 = {};
    upWind_thirdOrder_X(u2 ,v2, du3);
    upWind_thirdOrder_Y(u2 ,v2, dv3);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u_next[i][j] = 1.0 / 3.0 * u[i][j] + 2.0 / 3.0 * u2[i][j] + 2.0 / 3.0 * dt * du3[i][j];
            v_next[i][j] = 1.0 / 3.0 * v[i][j] + 2.0 / 3.0 * v2[i][j] + 2.0 / 3.0 * dt * dv3[i][j];
        }
    }
}

Array operator+(Array& u, Array& v){
    Array ans = {};
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            ans[i][j] = u[i][j] + v[i][j];
        }
    }
    return ans;
}

Array operator*(double a, Array& u){
    Array ans = {};
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            ans[i][j] = a * u[i][j];
        }
    }
    return ans;
}

void RK3_3rdOrder(Array& u, Array& v, Array& u_next, Array& v_next){
    Array du1 = {};
    Array dv1 = {};
    Array u1  = {};
    Array v1  = {};
    upWind_thirdOrder_X(u ,v, du1);
    upWind_thirdOrder_Y(u ,v, dv1);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u1[i][j] = dt * du1[i][j];
            v1[i][j] = dt * dv1[i][j];
        }
    }
    Array du2 = {};
    Array dv2 = {};
    Array u2  = {};
    Array v2  = {};
    upWind_thirdOrder_X(u1 , v1, du2);
    upWind_thirdOrder_Y(u1 , v1, dv2);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u2[i][j] = 1.0 / 2.0 * dt * du2[i][j];
            v2[i][j] = 1.0 / 2.0 * dt * dv2[i][j];
        }
    }
    Array du3 = {};
    Array dv3 = {};
    Array u3  = {};
    Array v3  = {};
    upWind_thirdOrder_X(u2 ,v2, du3);
    upWind_thirdOrder_Y(u2 ,v2, dv3);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u3[i][j] = - 1.0 * dt * du2[i][j] + 2.0 * dt * du3[i][j];
            v3[i][j] = - 1.0 * dt * dv2[i][j] + 2.0 * dt * dv3[i][j];
        }
    }
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u_next[i][j] = u[i][j] + 1.0 / 6.0 * u1[i][j] + 2.0 / 3.0 * u2[i][j] + 1.0 / 6.0 * u3[i][j];
            v_next[i][j] = v[i][j] + 1.0 / 6.0 * v1[i][j] + 2.0 / 3.0 * v2[i][j] + 1.0 / 6.0 * v3[i][j];
        }
    }
}

void RK4_3rdOrder(Array& u, Array& v, Array& u_next, Array& v_next){
    Array du1 = {};
    Array dv1 = {};
    Array u1  = {};
    Array v1  = {};
    upWind_thirdOrder_X(u ,v, du1);
    upWind_thirdOrder_Y(u ,v, dv1);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u1[i][j] = u[i][j] + dt * du1[i][j];
            v1[i][j] = v[i][j] + dt * dv1[i][j];
        }
    }
    Array du2 = {};
    Array dv2 = {};
    Array u2  = {};
    Array v2  = {};
    upWind_thirdOrder_X(u1 ,v1, du2);
    upWind_thirdOrder_Y(u1 ,v1, dv2);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u2[i][j] = u[i][j] + 1.0 / 2.0 * dt * du2[i][j];
            v2[i][j] = v[i][j] + 1.0 / 2.0 * dt * dv2[i][j];
        }
    }
    Array du3 = {};
    Array dv3 = {};
    Array u3  = {};
    Array v3  = {};
    upWind_thirdOrder_X(u2 ,v2, du3);
    upWind_thirdOrder_Y(u2 ,v2, dv3);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u3[i][j] = u[i][j] + 1.0 / 2.0 * dt * du3[i][j];
            v3[i][j] = v[i][j] + 1.0 / 2.0 * dt * dv3[i][j];
        }
    }
    Array du4 = {};
    Array dv4 = {};
    Array u4  = {};
    Array v4  = {};
    upWind_thirdOrder_X(u3 ,v3, du4);
    upWind_thirdOrder_Y(u3 ,v3, dv4);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u4[i][j] = u[i][j] + 1.0 / 2.0 * dt * du3[i][j];
            v4[i][j] = v[i][j] + 1.0 / 2.0 * dt * dv3[i][j];
        }
    }
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u_next[i][j] = 1.0 / 6.0 * u1[i][j] + 1.0 / 3.0 * u2[i][j] + 1.0 / 3.0 * u3[i][j] + 1.0 / 6.0 * u4[i][j];
            v_next[i][j] = 1.0 / 6.0 * v1[i][j] + 1.0 / 3.0 * v2[i][j] + 1.0 / 3.0 * v3[i][j] + 1.0 / 6.0 * v4[i][j];
        }
    }
}

void RK4_5rdOrder(Array& u, Array& v, Array& u_next, Array& v_next){
    Array du1 = {};
    Array dv1 = {};
    Array u1  = {};
    Array v1  = {};
    upWind_fifthOrder_X(u ,v, du1);
    upWind_fifthOrder_Y(u ,v, dv1);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u1[i][j] = u[i][j] + dt * du1[i][j];
            v1[i][j] = v[i][j] + dt * dv1[i][j];
        }
    }
    Array du2 = {};
    Array dv2 = {};
    Array u2  = {};
    Array v2  = {};
    upWind_fifthOrder_X(u1 ,v1, du2);
    upWind_fifthOrder_Y(u1 ,v1, dv2);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u2[i][j] = u[i][j] + 1.0 / 2.0 * dt * du2[i][j];
            v2[i][j] = v[i][j] + 1.0 / 2.0 * dt * dv2[i][j];
        }
    }
    Array du3 = {};
    Array dv3 = {};
    Array u3  = {};
    Array v3  = {};
    upWind_fifthOrder_X(u2 ,v2, du3);
    upWind_fifthOrder_Y(u2 ,v2, dv3);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u3[i][j] = u[i][j] + 1.0 / 2.0 * dt * du3[i][j];
            v3[i][j] = v[i][j] + 1.0 / 2.0 * dt * dv3[i][j];
        }
    }
    Array du4 = {};
    Array dv4 = {};
    Array u4  = {};
    Array v4  = {};
    upWind_fifthOrder_X(u3 ,v3, du4);
    upWind_fifthOrder_Y(u3 ,v3, dv4);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u4[i][j] = u[i][j] + 1.0 / 2.0 * dt * du3[i][j];
            v4[i][j] = v[i][j] + 1.0 / 2.0 * dt * dv3[i][j];
        }
    }
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u_next[i][j] = 1.0 / 6.0 * u1[i][j] + 1.0 / 3.0 * u2[i][j] + 1.0 / 3.0 * u3[i][j] + 1.0 / 6.0 * u4[i][j];
            v_next[i][j] = 1.0 / 6.0 * v1[i][j] + 1.0 / 3.0 * v2[i][j] + 1.0 / 3.0 * v3[i][j] + 1.0 / 6.0 * v4[i][j];
        }
    }
}

int main(){
    std::cout << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 - 8);

    Array u = {};
    Array v = {};
    Array u_next = {};
    Array v_next = {};
    double t = 0.0;

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
            coef = 2.0 * norm;
            fprintf(gp, "%f %f %f %f %f\n", i * dx , j * dy, coef * u[i][j] / (norm * Nx), coef * v[i][j] / (norm * Ny), norm);
        }
    }
    fprintf(gp, "e\n");
    fflush(gp);

    std::cout << "Enterキーを押してください．" << std::endl;
    getchar();

    for(int i=1; t<1.0; ++i){
        //Euler_1stOrder(u, v, u_next, v_next);
        //Euler_3rdOrder(u, v, u_next, v_next);
        //TVD_RK3_3rdOrder(u, v, u_next, v_next);
        RK3_3rdOrder(u, v, u_next, v_next);
        //RK4_3rdOrder(u, v, u_next, v_next);
        //RK4_5rdOrder(u, v, u_next, v_next);
        clear(u, v, u_next, v_next);

        // 描画
        if(i%INTV == 0){
            fprintf(gp, "plot '-' with vector lc palette\n");
            for(int i=0; i<Nx; ++i){
                for(int j=0; j<Ny; ++j){
                    norm = std::sqrt(u[i][j] * u[i][j] + v[i][j] * v[i][j]);
                    coef = 2.0 * norm;
                    fprintf(gp, "%f %f %f %f %f\n", i * dx , j * dy, coef * u[i][j] / (norm * Nx), coef * v[i][j] / (norm * Ny), norm);
                }
            }
            fprintf(gp, "e\n");
            fflush(gp);
            std::cout << t << std::endl;
        }
        t = i * dt;
    }
}