/*
 * 移流のテスト
 * 
 * NS方程式
 */

#include <iostream>
#include <array>
#include <cmath>
#include <limits>
#include <iomanip>
#include <algorithm>
#include <omp.h>

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
const int INTV     = 1000;
const double mu    = 1.83e-5;
double dt          = std::min(0.1 / (std::sqrt(a * a + b * b) * std::sqrt((1.0 / dx) * (1.0 / dx) + (1.0 / dy) * (1.0 / dy)))
                            , 0.1 * mu * 0.5 / (1.0 / (dx * dx) + 1.0 / (dy * dy)));
const int view     = 0;
const double R     = 8.31446261815324; // J K^1 mol^1
const double Air   = 28.966; // g/mol
const double T     = 273;    // K
const double air0  = 1293;   // g/m^3

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
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            px_next[i][j] = ( - px[i+2][j] + 8.0 * px[i+1][j] - 8.0 * px[i-1][j] + px[i-2][j]) / (12.0 * dx);
        }
    }
}

void d_Y(Array& py, Array& py_next){
    #pragma omp parallel for
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
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u_next[i][j] = mu * (
                ( - u[i-2][j] + 16.0 * u[i-1][j] - 30 * u[i][j] + 16.0 * u[i+1][j] - u[i+2][j]) / (12.0 * dx * dx)
                +
                ( - u[i][j-2] + 16.0 * u[i][j-1] - 30 * u[i][j] + 16.0 * u[i][j+1] - u[i][j+2]) / (12.0 * dx * dx));
        }
    }
}

void diffusion_fourthOrder_Y(Array& v, Array& v_next){
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            v_next[i][j] = mu * (
                ( - v[i-2][j] + 16.0 * v[i-1][j] - 30 * v[i][j] + 16.0 * v[i+1][j] - v[i+2][j]) / (12.0 * dx * dx)
                +
                ( - v[i][j-2] + 16.0 * v[i][j-1] - 30 * v[i][j] + 16.0 * v[i][j+1] - v[i][j+2]) / (12.0 * dx * dx));
        }
    }
}

/********************************/
/*      up wind 5th scheme      */
/********************************/

void upWind_fifthOrder_X(Array& u, Array& v, Array& u_next){
    double ad1, ad2;
    #pragma omp parallel for
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
    #pragma omp parallel for
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

// NS用のリー微分になっているので注意．
void upWind_fifthOrder_Scalar(Array& u, Array& v, Array& rho, Array& rho_next){
    double ad1, ad2;
    Array ux = {};
    Array vy = {};
    d_X(u, ux);
    d_Y(v, vy);
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            ad1 = u[i][j];
            ad2 = v[i][j];
            rho_next[i][j] =
                - ( ad1 * ( rho[i+3][j] - 9.0 * rho[i+2][j] + 45.0 * (rho[i+1][j] - rho[i-1][j]) + 9.0 * rho[i-2][j] - rho[i-3][j] ) / (60.0 * dx)
                    + std::abs(ad1) * ( - rho[i+3][j] + 6.0 * rho[i+2][j] - 15.0 * rho[i+1][j] + 20.0 * rho[i][j] - 15.0 * rho[i-1][j] + 6.0 * rho[i-2][j] - rho[i-3][j] ) / (60.0 * dx))
                - ( ad2 * ( rho[i][j+3] - 9.0 * rho[i][j+2] + 45.0 * (rho[i][j+1] - rho[i][j-1]) + 9.0 * rho[i][j-2] - rho[i][j-3] ) / (60.0 * dy)
                    + std::abs(ad2) * ( - rho[i][j+3] + 6.0 * rho[i][j+2] - 15.0 * rho[i][j+1] + 20.0 * rho[i][j] - 15.0 * rho[i][j-1] + 6.0 * rho[i][j-2] - rho[i][j-3] ) / (60.0 * dy))
                - rho[i][j] * (ux[i][j] + vy[i][j]);
        }
    }
}

// clear
void clear(Array& u, Array& v, Array& u_next, Array& v_next){
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u[i][j] = u_next[i][j];
            v[i][j] = v_next[i][j];
        }
    }
}

void clear_rho(Array& rho, Array& rho_next){
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            rho[i][j] = rho_next[i][j];
        }
    }
}

// initialize function
void init_func(Array& u, Array& v){
    double x = 0.0;
    double y = 0.0;
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            x = i * dx;
            y = j * dy;
            if(y <= 0.5){
                u[i][j] =   1.0;
                v[i][j] =   0.0;
            } else if ( y >= 0.5){
                u[i][j] = - 0.5;
                v[i][j] =   0.0;
            }
        }
    }
    u[Nx/2][Ny/2] = 0.5;
    v[Nx/2][Ny/2] = 0.5;
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u[i][j] =   0.0;
            v[i][j] =   0.0;
	}
    }
}

void make_wind(Array& u, Array& v){
    double x = 0.0;
    double y = 0.0;
    double a = 0.01;
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
	x = i * dx;
        u[i][0]    =   a * std::sin(pi * x);
        u[i][Ny-1] = - a * std::sin(pi * x);
    }
    #pragma omp parallel for
    for(int j=0; j<Ny; ++j){
	y = j * dy;
        v[0][j]    = - a * std::sin(pi * y);
        v[Nx-1][j] =   a * std::sin(pi * y);
    }
}

// initialize rho function
void init_func_rho(Array& rho){
    double x = 0.0;
    double y = 0.0;
    double a = Lx / 2.0;
    double b = Ly / 2.0;
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            x = i * dx;
            y = j * dy;
            rho[i][j] = air0 + 0.01 * air0 * std::exp( - 100.0 * ((x - a) * (x - a) + (y - b) * (y - b)));
	    /*
	    if(y >= 0.5){
                rho[i][j] = 11.0; // + 1.0 * std::exp( - 100.0 * ((x - a) * (x - a) + (y - b) * (y - b)));
	    } else {
    	        rho[i][j] = 10.0; // + 1.0 * std::exp( - 100.0 * ((x - a) * (x - a) + (y - b) * (y - b)));
            }
	    */
    	}
    }
}

Array operator+(Array& u, Array& v){
    Array ans = {};
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            ans[i][j] = u[i][j] + v[i][j];
        }
    }
    return ans;
}

Array operator*(double a, Array& u){
    Array ans = {};
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            ans[i][j] = a * u[i][j];
        }
    }
    return ans;
}

void eqState(Array& rho, Array& p){
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            p[i][j] = rho[i][j] / Air * R * T;
        }
    } 
}

// Schemes
void NS_RK4_5rdOrder(Array& u, Array& v, Array& rho, Array& u_next, Array& v_next, Array& rho_next){
    Array du1   = {};
    Array dv1   = {};
    Array u1    = {};
    Array v1    = {};
    Array difu1 = {};
    Array difv1 = {};
    Array rho1  = {};
    Array drho1 = {};
    Array p1    = {};
    Array px1   = {};
    Array py1   = {};
    upWind_fifthOrder_X(u ,v, du1);
    upWind_fifthOrder_Y(u ,v, dv1);
    diffusion_fourthOrder_X(u, difu1);
    diffusion_fourthOrder_Y(v, difv1);
    upWind_fifthOrder_Scalar(u, v, rho, drho1);
    eqState(rho, p1);
    d_X(p1, px1);
    d_Y(p1, py1);
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u1[i][j] = u[i][j] + dt * (du1[i][j] + (difu1[i][j] - px1[i][j]) / rho[i][j]);
            v1[i][j] = v[i][j] + dt * (dv1[i][j] + (difv1[i][j] - py1[i][j]) / rho[i][j]);
            rho1[i][j] = rho[i][j] + dt * drho1[i][j];
        }
    }
    Array du2   = {};
    Array dv2   = {};
    Array u2    = {};
    Array v2    = {};
    Array difu2 = {};
    Array difv2 = {};
    Array rho2  = {};
    Array drho2 = {};
    Array p2    = {};
    Array px2   = {};
    Array py2   = {};
    upWind_fifthOrder_X(u1 ,v1, du2);
    upWind_fifthOrder_Y(u1 ,v1, dv2);
    diffusion_fourthOrder_X(u1, difu2);
    diffusion_fourthOrder_Y(v1, difv2);
    upWind_fifthOrder_Scalar(u1, v1, rho1, drho2);
    eqState(rho1, p2);
    d_X(p2, px2);
    d_Y(p2, py2);
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u2[i][j] = u[i][j] + 1.0 / 2.0 * dt * (du2[i][j] + (difu2[i][j] - px2[i][j]) / rho[i][j]);
            v2[i][j] = v[i][j] + 1.0 / 2.0 * dt * (dv2[i][j] + (difv2[i][j] - py2[i][j]) / rho[i][j]);
            rho2[i][j] = rho[i][j] + 1.0 / 2.0 * dt * drho2[i][j];
        }
    }
    Array du3   = {};
    Array dv3   = {};
    Array u3    = {};
    Array v3    = {};
    Array difu3 = {};
    Array difv3 = {};
    Array rho3  = {};
    Array drho3 = {};
    Array p3    = {};
    Array px3   = {};
    Array py3   = {};
    upWind_fifthOrder_X(u2 ,v2, du3);
    upWind_fifthOrder_Y(u2 ,v2, dv3);
    diffusion_fourthOrder_X(u2, difu3);
    diffusion_fourthOrder_Y(v2, difv3);
    upWind_fifthOrder_Scalar(u2, v2, rho2, drho3);
    eqState(rho2, p3);
    d_X(p3, px3);
    d_Y(p3, py3);
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u3[i][j] = u[i][j] + 1.0 / 2.0 * dt * (du3[i][j] + (difu3[i][j] - px3[i][j]) / rho[i][j]);
            v3[i][j] = v[i][j] + 1.0 / 2.0 * dt * (dv3[i][j] + (difv3[i][j] - py3[i][j]) / rho[i][j]);
            rho3[i][j] = rho[i][j] + 1.0 / 2.0 * dt * drho3[i][j];
        }
    }
    Array du4   = {};
    Array dv4   = {};
    Array u4    = {};
    Array v4    = {};
    Array difu4 = {};
    Array difv4 = {};
    Array rho4  = {};
    Array drho4 = {};
    Array p4    = {};
    Array px4   = {};
    Array py4   = {};
    upWind_fifthOrder_X(u3 ,v3, du4);
    upWind_fifthOrder_Y(u3 ,v3, dv4);
    diffusion_fourthOrder_X(u3, difu4);
    diffusion_fourthOrder_Y(v3, difv4);
    upWind_fifthOrder_Scalar(u3, v3, rho3, drho4);
    eqState(rho3, p4);
    d_X(p4, px4);
    d_Y(p4, py4);
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            du4[i][j] = (du4[i][j] + (difu4[i][j] - px4[i][j]) / rho[i][j]);
            dv4[i][j] = (dv4[i][j] + (difv4[i][j] - py4[i][j]) / rho[i][j]);
            drho4[i][j] = drho4[i][j];
        }
    }
    #pragma omp parallel for
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            u_next[i][j] = u[i][j] + 1.0 / 6.0 * dt * (du1[i][j] + 2.0 * du2[i][j] + 2.0 * du3[i][j] + du4[i][j]);
            v_next[i][j] = v[i][j] + 1.0 / 6.0 * dt * (dv1[i][j] + 2.0 * dv2[i][j] + 2.0 * dv3[i][j] + dv4[i][j]);
            rho_next[i][j] = rho[i][j] + 1.0 / 6.0 * dt * (drho1[i][j] + 2.0 * drho2[i][j] + 2.0 * drho3[i][j] + drho4[i][j]);
        }
    }
}

int main(){
    std::cout << std::fixed << std::setprecision(std::numeric_limits<double>::digits10);

    Array u        = {};
    Array v        = {};
    Array rho      = {};
    Array u_next   = {};
    Array v_next   = {};
    Array rho_next = {};
    double t = 0.0;

    init_func(u, v);
    init_func_rho(rho);

    /**********************************************************************/
    /*                 可視化の設定(gnuplot)                                */
    /**********************************************************************/
    double norm = 0.0;
    double coef = 1.0;
    std::FILE *gp = popen( "gnuplot -persist", "w" );
    if(view == 0){
        fprintf(gp, "set xr [0:%f]\n", Lx);
        fprintf(gp, "set yr [0:%f]\n", Ly);
        fprintf(gp, "set contour\n");
        fprintf(gp, "set grid\n");
        fprintf(gp, "unset key\n");
        fprintf(gp, "set size ratio 1\n");
        fprintf(gp, "set palette rgb 33,13,10\n");
        fprintf(gp, "set term png size 1080, 1080\n");

        //初期条件描画
        fprintf(gp, "plot '-' with vector lc palette\n");
        fprintf(gp, "splot '-' w l\n");
        for(int i=0; i<Nx; ++i){
            for(int j=0; j<Ny; ++j){
                norm = std::sqrt(u[i][j] * u[i][j] + v[i][j] * v[i][j]);
                fprintf(gp, "%lf %lf %lf %lf %lf\n", i * dx , j * dy, coef * u[i][j] / (norm * Nx), coef * v[i][j] / (norm * Ny), norm);
                //fprintf(gp, "%lf %lf %lf\n", i * dx , j * dy, rho[i][j]);
            }
            //fprintf(gp, "\n");
        }
        fprintf(gp, "e\n");
        fflush(gp);

        // std::cout << "Enterキーを押してください．" << std::endl;
        // getchar();
    }

    int g = 1;
    for(int i=0; t<100000.0; ++i){
    	// make_wind(u, v);
        NS_RK4_5rdOrder(u, v, rho, u_next, v_next, rho_next);
        clear(u, v, u_next, v_next);
        double rho_max = 0.0;
        double int_rho = 0.0;
        double u_max = 0.0;
        double v_max = 0.0;
        for(int i=0; i<Nx; ++i){
            for(int j=0; j<Ny; ++j){
                u_max = std::max(u_max, std::abs(u[i][j]));
                v_max = std::max(v_max, std::abs(v[i][j]));
                rho_max = std::max(rho_max, std::abs(rho[i][j]));
                int_rho += rho[i][j] * dx * dy;
            }
        }

        double co = 0.9;
        dt = std::min(co / (std::sqrt(u_max * u_max + v_max * v_max) * std::sqrt((1.0 / dx) * (1.0 / dx) + (1.0 / dy) * (1.0 / dy)))
                    , co * (mu / rho_max) * 0.5 / (1.0 / (dx * dx) + 1.0 / (dy * dy)));
        
        // 描画
        if(i%INTV == 0 && view == 0){
            fprintf(gp, "plot '-' with vector lc palette\n");
            //fprintf(gp, "splot '-' w l\n");
            int step = 1;
	        coef = 1.2;
            for(int i=0; i<Nx; i += step){
                for(int j=0; j<Ny; j += step){
                    norm = std::sqrt(u[i][j] * u[i][j] + v[i][j] * v[i][j]);
                    fprintf(gp, "%.16f %.16f %.16f %.16f %.16f\n", i * dx , j * dy, coef * u[i][j] / (norm * Nx), coef * v[i][j] / (norm * Ny), norm);
                    //fprintf(gp, "%lf %lf %lf\n", i * dx , j * dy, rho[i][j]);
                }
                //fprintf(gp, "\n");
            }
            fprintf(gp, "e\n");
            fflush(gp);
	        fprintf(gp, "set output 'air%06d.png'\n", g);
	        g++;
        }
        // 出力
        /*
        if(i%INTV == 0){
            for(int i=0; i<Nx; ++i){
                for(int j=0; j<Ny; ++j){
                    std::cout << i * dx << " " << j * dy << " " << u[i][j] << " " << v[i][j] << " " << rho[i][j] << std::endl;
                }
            }
	        std::cout << std::endl;
        }
        */
        t += dt;
        if(i%INTV == 0){
            std::cout << "time: " << t << " dt: " << dt << " u_max: " << u_max << " v_max: " << v_max << " rho_max: " << rho_max << " int_rho: " << int_rho
                      << " P: " << rho_max / Air * R * T << std::endl;
        }
    }
    pclose(gp);
}
