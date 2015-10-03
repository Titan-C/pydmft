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

    std::vector<double> x(N);
    std::copy (g + k*N, g + (k+1)*N, x.begin());//column fortran
    x[k] -= 1;

    std::vector<double> y(N);

    for(unsigned int i=0; i<N; i++)
        y[i] = g[i*N + k];//row fortran

    cblas_dger (CblasColMajor, N, N, a, &x[0], 1, &y[0], 1, &g[0], N);

    x.clear();
    y.clear();

}

void cg2flip(size_t N, double *g, double *dv, int l, int k){
  std::vector<double> id2 (4, 0.);
  id2[0] = id2[3] = 1.;

  for(auto ent: id2) std::cout << ent << "\n";
  std::vector<double> U (2*N);
  std::copy (g + k*N, g + (k+1)*N, U.begin());//column fortran
  for(auto ent: U) std::cout << ent << " ";
  std::vector<double> V (2*N);


}
