以下のように定める．（定義不足だが，適宜うまく読んで下さい．）
$$
\vec{u} = [u, v]^T = [u(x,y), v(x,y)]^T \\
\vec{c} = [a, b]^T = \rm{const. (vector\ field)} \\
$$
ベクトル場 [a, b]^T に沿う [u, v]^T の移流方程式は，
$$
\dfrac{\partial \vec{u}}{\partial t} + \left( \vec{c} \cdot \left[\dfrac{\partial}{\partial x}, \dfrac{\partial}{\partial y}\right]^T \right) \vec{u} = 0 \\
\dfrac{\partial \vec{u}}{\partial t} + \left( a \dfrac{\partial}{\partial x} + b\dfrac{\partial}{\partial y} \right) \vec{u} = 0 \\
\dfrac{\partial \vec{u}}{\partial t} + a \dfrac{\partial \vec{u}}{\partial x} + b\dfrac{\partial \vec{u}}{\partial y} = 0 \\
\left[ \dfrac{\partial u}{\partial t} ,\dfrac{\partial v}{\partial t} \right]^T + a \left[ \dfrac{\partial u}{\partial x} ,\dfrac{\partial v}{\partial x} \right]^T + b \left[ \dfrac{\partial u}{\partial y} ,\dfrac{\partial v}{\partial y} \right]^T = 0
$$
なので，成分ごとに書くと，
$$
\dfrac{\partial u}{\partial t} + a \dfrac{\partial u}{\partial x} + b \dfrac{\partial u}{\partial y} = 0 \\
\dfrac{\partial v}{\partial t} + a \dfrac{\partial v}{\partial x} + b \dfrac{\partial v}{\partial y} = 0
$$
これを差分化（1次の風上差分法）すると，（vについては省略．）
$$
\dfrac{u^{n+1}_{i,j} - u^{n}_{i,j}}{\Delta t} + a \dfrac{u^{n}_{i+1,j} - u^{n}_{i-1,j}}{2\Delta x} + |a| \dfrac{-u^{n}_{i+1,j} + 2u^{n}_{i,j} - u^{n}_{i-1,j}}{2\Delta x} \\ + b \dfrac{u^{n}_{i,j+1} - u^{n}_{i,j-1}}{2\Delta y} + |b| \dfrac{-u^{n}_{i,j+1} + 2u^{n}_{i,j} - u^{n}_{i,j-1}}{2\Delta y} = 0
$$
となる．（非粘性）バーガース方程式については，
$$
\vec{c} := \vec{u}
$$
とすれば良いので，（2次元）バーガース方程式についての差分法は，
$$
\dfrac{u^{n+1}_{i,j} - u^{n}_{i,j}}{\Delta t} + u^{n}_{i,j} \dfrac{u^{n}_{i+1,j} - u^{n}_{i-1,j}}{2\Delta x} + |u^{n}_{i,j}| \dfrac{-u^{n}_{i+1,j} + 2u^{n}_{i,j} - u^{n}_{i-1,j}}{2\Delta x} \\ 
+ v^{n}_{i,j} \dfrac{u^{n}_{i,j+1} - u^{n}_{i,j-1}}{2\Delta y} + |v^{n}_{i,j}| \dfrac{-u^{n}_{i,j+1} + 2u^{n}_{i,j} - u^{n}_{i,j-1}}{2\Delta y} = 0
$$
となる．（vの時間発展も同様の差分を取れば良い．）

コードに落とし込むと，移流の方は，c = [ad1, ad2]^T = [1, 1]と見れば，

```C++
void upWind_firstOrder_X(Array& u, Array& v, Array& u_next){
    double ad1, ad2;
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            ad1 = 1.0;
            ad2 = 1.0;
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
            ad1 = 1.0;
            ad2 = 1.0;
            v_next[i][j] = v[i][j]
                            - dt * ( ad1 * (v[i+1][j] - v[i-1][j]) / (2.0 * dx) + std::abs(ad1) * (- v[i+1][j] + 2.0 * v[i][j] - v[i-1][j]) / (2.0 * dx))
                            - dt * ( ad2 * (v[i][j+1] - v[i][j-1]) / (2.0 * dy) + std::abs(ad2) * (- v[i][j+1] + 2.0 * v[i][j] - v[i][j-1]) / (2.0 * dy));
        }
    }
}
```

バーガース方程式の方は，

```C++
void upWind_firstOrder_X(Array& u, Array& v, Array& u_next){
    double ad1, ad2;
    for(int i=0; i<Nx; ++i){
        for(int j=0; j<Ny; ++j){
            ad1 = u[i][j];
            ad2 = v[i][j];
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
            ad1 = u[i][j];
            ad2 = v[i][j];
            v_next[i][j] = v[i][j]
                            - dt * ( ad1 * (v[i+1][j] - v[i-1][j]) / (2.0 * dx) + std::abs(ad1) * (- v[i+1][j] + 2.0 * v[i][j] - v[i-1][j]) / (2.0 * dx))
                            - dt * ( ad2 * (v[i][j+1] - v[i][j-1]) / (2.0 * dy) + std::abs(ad2) * (- v[i][j+1] + 2.0 * v[i][j] - v[i][j-1]) / (2.0 * dy));
        }
    }
}
```

となる．



by mino2357