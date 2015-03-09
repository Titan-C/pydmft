#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash

#$ -q theo-ox.q
#$ -pe mpi 12
#$ -N oscardmft


MPI_DIR=/opt/openmpi


export PATH=~/libs/bin:$PATH
export LD_LIBRARY_PATH=/home/oscar/libs/lib:$LD_LIBRARY_PATH

export PATH=~/miniconda/bin:$PATH

source activate alps
$MPI_DIR/bin/mpirun -np $NSLOTS python $1
