// memo


const double alpha = 1.0;   // 1:UTOPIA 3:K-K

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