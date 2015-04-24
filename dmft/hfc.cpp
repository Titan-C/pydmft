/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2015  Óscar Nájera <najera.oscar@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "hfc.h"

void cgnew(size_t N, double *g, double dv, int k){
    double ee, a;

    ee = exp(dv)-1.;
    a = ee/(1. + (1.-g[k*N + k])*ee);

    std::vector<double> x, y;
    x.resize(N);
    y.resize(N);

    for(unsigned int i=0; i<N; i++){
        x[i] = g[k*N +i];//column fortran
        y[i] = g[i*N +k];//row fortran
    }
    x[k] -= 1;

    cblas_dger (CblasColMajor, N, N, a, &x[0], 1, &y[0], 1, &g[0], N);

    x.clear();
    y.clear();

}