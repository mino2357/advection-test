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
#include <algorithm>

// parameters
const double Lx    = 1.0;
const double Ly    = 1.0;
const int Nx       = 32;
const int Ny       = 32;
const double dx    = Lx / (Nx - 1);
const double dy    = Ly / (Ny - 1);
const double a     = 1.0; // advection [a, b]^T
const double b     = 1.0; // vector field
const double pi    = std::acos(-1.0);
const double alpha = 1.0;   // 1:UTOPIA 3:K-K
const int INTV     = 10;
const double Re    = 1;
const double dt    = std::min(0.01 / (std::sqrt(a * a + b * b) * std::sqrt((1.0 / dx) * (1.0 / dx) + (1.0 / dy) * (1.0 / dy)))
                              , 0.01 * Re * 0.5 / (1.0 / (dx * dx) + 1.0 / (dy * dy)));

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
/*   4 th order Central         */
/********************************/

void d_X(Array& px, Array& px_next){
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            px_next[i][j] = ( - px[i+2][j] + 8.0 * px[i+1][j] - 8.0 * px[i-1][j] + px[i-2][j]) / (12.0 * dx);
        }
    }
}

void d_Y(Array& py, Array& py_next){
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            py_next[i][j] = ( - py[i][j+2] + 8.0 * py[i][j+1] - 8.0 * py[i][j-1] + py[i][j-2]) / (12.0 * dy);
        }
    }
}

/********************************/
/*       diffusion              */
/********************************/

void diffusion_fourthOrder_X(Array& u, Array& u_next){
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u_next[i][j] = 1.0 / Re * (
                ( - u[i-2][j] + 16.0 * u[i-1][j] - 30 * u[i][j] + 16.0 * u[i+1][j] - u[i+2][j]) / (12.0 * dx * dx)
                +
                ( - u[i][j-2] + 16.0 * u[i][j-1] - 30 * u[i][j] + 16.0 * u[i][j+1] - u[i][j+2]) / (12.0 * dx * dx));
        }
    }
}

void diffusion_fourthOrder_Y(Array& v, Array& v_next){
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            v_next[i][j] = 1.0 / Re * (
                ( - v[i-2][j] + 16.0 * v[i-1][j] - 30 * v[i][j] + 16.0 * v[i+1][j] - v[i+2][j]) / (12.0 * dx * dx)
                +
                ( - v[i][j-2] + 16.0 * v[i][j-1] - 30 * v[i][j] + 16.0 * v[i][j+1] - v[i][j+2]) / (12.0 * dx * dx));
        }
    }
}

/********************************/
/*  up wind 1 th order scheme   */
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
            ad1 = u[i][j]; // - (j * dy - Ly/2.0); // u[i][j];
            ad2 = v[i][j]; //   (i * dx - Lx/2.0); // v[i][j];
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
            ad1 = u[i][j]; // - (j * dy - Ly/2.0); // u[i][j];
            ad2 = v[i][j]; //   (i * dx - Lx/2.0); // v[i][j];
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
            /*
            if(i>3*Nx/8 && i<5*Nx/8 && j>Ny/4 && j<3*Ny/4){
                u[i][j] = 1.0;
                v[i][j] = 1.0;
            }
            */
            u[i][j] = 0.0;
            v[i][j] = 0.0;
        }
    }
}

// initialize p function
void init_func_p(Array& p){
    double x = 0.0;
    double y = 0.0;
    double a = Lx / 2.0;
    double b = Ly / 2.0;
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            x = i * dx;
            y = j * dy;
            p[i][j] = std::exp( - 10.0 * ((x - a) * (x - a) + (y - b) * (y - b)));
        }
    }
}

void pressure(Array& p, double t){
    double x = 0.0;
    double y = 0.0;
    double s = 10.0;
    double r = 0.4;
    double a = Lx / 2.0;
    double b = Ly / 2.0;
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            x = i * dx - Lx / 2.0;
            y = j * dy - Ly / 2.0;
            p[i][j] = 0.01 * std::exp( - 10.0 * ((x - r * cos(0.8 * s * t) + 0.1) * (x - r * cos(0.8 * s * t) + 0.1) + (y - r * sin(s * t)) * (y - r * sin(s * t)))); // sin(2.0 * pi * (x * y) / Lx - 0.1 * t);
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

void Euler_5thOrder(Array& u, Array& v, Array& u_next, Array& v_next){
    Array du = {};
    Array dv = {};
    upWind_fifthOrder_X(u, v, du);
    upWind_fifthOrder_Y(u, v, dv);
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

void TVD_RK3_5thOrder(Array& u, Array& v, Array& u_next, Array& v_next){
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
            u2[i][j] = 3.0 / 4.0 * u[i][j] + 1.0 / 4.0 * u1[i][j] + 1.0 / 4.0 * dt * du2[i][j];
            v2[i][j] = 3.0 / 4.0 * v[i][j] + 1.0 / 4.0 * v1[i][j] + 1.0 / 4.0 * dt * dv2[i][j];
        }
    }
    Array du3 = {};
    Array dv3 = {};
    upWind_fifthOrder_X(u2 ,v2, du3);
    upWind_fifthOrder_Y(u2 ,v2, dv3);
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
            du3[i][j] = - 1.0 * dt * du2[i][j] + 2.0 * dt * du3[i][j];
            dv3[i][j] = - 1.0 * dt * dv2[i][j] + 2.0 * dt * dv3[i][j];
        }
    }
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u_next[i][j] = u[i][j] + 1.0 / 6.0 * du1[i][j] + 2.0 / 3.0 * du2[i][j] + 1.0 / 6.0 * du3[i][j];
            v_next[i][j] = v[i][j] + 1.0 / 6.0 * dv1[i][j] + 2.0 / 3.0 * dv2[i][j] + 1.0 / 6.0 * dv3[i][j];
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
            u_next[i][j] = u[i][j] + 1.0 / 6.0 * dt * (1.0 * u1[i][j] + 2.0 * u2[i][j] + 2.0 * u3[i][j] + 1.0 * u4[i][j]);
            v_next[i][j] = v[i][j] + 1.0 / 6.0 * dt * (1.0 * v1[i][j] + 2.0 * v2[i][j] + 2.0 * v3[i][j] + 1.0 * v4[i][j]);
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
            u_next[i][j] = u[i][j] + 1.0 / 6.0 * dt * (du1[i][j] + 2.0 * du2[i][j] + 2.0 * du3[i][j] + 1.0 * du4[i][j]);
            v_next[i][j] = v[i][j] + 1.0 / 6.0 * dt * (dv1[i][j] + 2.0 * dv2[i][j] + 2.0 * dv3[i][j] + 1.0 * dv4[i][j]);
        }
    }
}

void NS_RK4_5rdOrder(Array& u, Array& v, Array& p, Array& u_next, Array& v_next, Array& p_next){
    Array du1   = {};
    Array dv1   = {};
    Array u1    = {};
    Array v1    = {};
    Array difu1 = {};
    Array difv1 = {};
    Array dpx   = {};
    Array dpy   = {};
    d_X(p, dpx);
    d_Y(p, dpy);
    upWind_fifthOrder_X(u ,v, du1);
    upWind_fifthOrder_Y(u ,v, dv1);
    diffusion_fourthOrder_X(u, difu1);
    diffusion_fourthOrder_Y(v, difv1);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u1[i][j] = u[i][j] + dt * (du1[i][j] + difu1[i][j]);
            v1[i][j] = v[i][j] + dt * (dv1[i][j] + difv1[i][j]);
        }
    }
    Array du2   = {};
    Array dv2   = {};
    Array u2    = {};
    Array v2    = {};
    Array difu2 = {};
    Array difv2 = {};
    upWind_fifthOrder_X(u1 ,v1, du2);
    upWind_fifthOrder_Y(u1 ,v1, dv2);
    diffusion_fourthOrder_X(u1, difu2);
    diffusion_fourthOrder_Y(v1, difv2);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u2[i][j] = u[i][j] + 1.0 / 2.0 * dt * (du2[i][j] + difu2[i][j]);
            v2[i][j] = v[i][j] + 1.0 / 2.0 * dt * (dv2[i][j] + difv2[i][j]);
        }
    }
    Array du3   = {};
    Array dv3   = {};
    Array u3    = {};
    Array v3    = {};
    Array difu3 = {};
    Array difv3 = {};
    upWind_fifthOrder_X(u2 ,v2, du3);
    upWind_fifthOrder_Y(u2 ,v2, dv3);
    diffusion_fourthOrder_X(u2, difu3);
    diffusion_fourthOrder_Y(v2, difv3);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u3[i][j] = u[i][j] + 1.0 / 2.0 * dt * (du3[i][j] + difu3[i][j]);
            v3[i][j] = v[i][j] + 1.0 / 2.0 * dt * (dv3[i][j] + difv3[i][j]);
        }
    }
    Array du4   = {};
    Array dv4   = {};
    Array u4    = {};
    Array v4    = {};
    Array difu4 = {};
    Array difv4 = {};
    upWind_fifthOrder_X(u3 ,v3, du4);
    upWind_fifthOrder_Y(u3 ,v3, dv4);
    diffusion_fourthOrder_X(u3, difu4);
    diffusion_fourthOrder_Y(v3, difv4);
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            du4[i][j] = (du4[i][j] + difu4[i][j]);
            dv4[i][j] = (dv4[i][j] + difv4[i][j]);
        }
    }
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u_next[i][j] = u[i][j] + 1.0 / 6.0 * dt * (du1[i][j] + 2.0 * du2[i][j] + 2.0 * du3[i][j] + 1.0 * du4[i][j]) - dt * dpx[i][j];
            v_next[i][j] = v[i][j] + 1.0 / 6.0 * dt * (dv1[i][j] + 2.0 * dv2[i][j] + 2.0 * dv3[i][j] + 1.0 * dv4[i][j]) - dt * dpy[i][j];
        }
    }
}

int main(){
    std::cout << std::fixed << std::setprecision(std::numeric_limits<double>::digits10);

    Array u      = {};
    Array v      = {};
    Array p      = {};
    Array u_next = {};
    Array v_next = {};
    Array p_next = {};
    double t = 0.0;

    init_func(u, v);
    init_func_p(p);

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
    double coef = 1.0;
    fprintf(gp, "plot '-' with vector lc palette\n");
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            norm = std::sqrt(u[i][j] * u[i][j] + v[i][j] * v[i][j]);
            //coef = 2.0 * norm;
            fprintf(gp, "%f %f %f %f %f\n", i * dx , j * dy, coef * u[i][j] / (norm * Nx), coef * v[i][j] / (norm * Ny), norm);
        }
    }
    fprintf(gp, "e\n");
    fflush(gp);

    //std::cout << "Enterキーを押してください．" << std::endl;
    //getchar();

    for(int i=1; t<100.0; ++i){
        //Euler_1stOrder(u, v, u_next, v_next);
        //Euler_3rdOrder(u, v, u_next, v_next);
        //Euler_5thOrder(u, v, u_next, v_next);
        //TVD_RK3_3rdOrder(u, v, u_next, v_next);
        //TVD_RK3_5thOrder(u, v, u_next, v_next);
        //RK3_3rdOrder(u, v, u_next, v_next);
        //RK4_3rdOrder(u, v, u_next, v_next);
        //RK4_5rdOrder(u, v, u_next, v_next);
        NS_RK4_5rdOrder(u, v, p, u_next, v_next, p_next);
        clear(u, v, u_next, v_next);

        // 描画
        if(i%INTV == 0){
            fprintf(gp, "plot '-' with vector lc palette\n");
            int step = 1;
            for(int i=0; i<Nx; i += step){
                for(int j=0; j<Ny; j += step){
                    norm = std::sqrt(u[i][j] * u[i][j] + v[i][j] * v[i][j]);
                    //coef = 2.0 * norm;
                    fprintf(gp, "%f %f %f %f %f\n", i * dx , j * dy, coef * u[i][j] / (norm * Nx), coef * v[i][j] / (norm * Ny), norm);
                }
            }
            fprintf(gp, "e\n");
            fflush(gp);
            std::cout << t << std::endl;
        }
        t = i * dt;
        pressure(p, t);
    }
}
