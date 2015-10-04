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
  std::valarray<double> id2 (0., 4);
  id2[0] = id2[3] = 1.;

  std::valarray<double> U (2*N);
  std::copy (g + l*N, g + (l+1)*N, std::begin(U));//column fortran
  std::copy (g + k*N, g + (k+1)*N, std::begin(U) + N);//column fortran
  U[l] -= 1.;
  U[N+k] -= 1.;

  U[std::slice(0, N, 1)] *= std::valarray<double>(exp(dv[0])-1., N);
  U[std::slice(N, N, 1)] *= std::valarray<double>(exp(dv[1])-1., N);

  std::valarray<double> V (2*N);
  int col=0;
  for(size_t i=0; i<N; i++){
      V[i*2] = g[i*N + l];
      V[i*2+1] = g[i*N + k];
  }

  size_t sel[] = {l, k, l+N, k+N};
  std::valarray<size_t> myselection (sel,4);
  std::valarray<double> mat (U[myselection]);
  mat -= id2;
  int n = 2, info;
  std::valarray<int> ipiv(n);
  info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, N, &mat[0], n, &ipiv[0], &V[0], n);
  cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans,
	       N, N, n, -1, &U[0], N, &V[0], n, 1., &g[0], N);
}
