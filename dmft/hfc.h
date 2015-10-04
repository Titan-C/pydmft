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

#ifndef HFC_H
#define HFC_H

#include <cstdlib>
#include <vector>
#include <valarray>
#include <iostream>
#include <cmath>

#include <cblas.h>
#include <lapacke.h>


void cgnew(size_t N, double *g, double dv, int k);
void cg2flip(size_t N, double *g, double *dv, int l, int k);

#endif // HFC_H
