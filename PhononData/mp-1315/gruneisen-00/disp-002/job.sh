#!/bin/bash
#QSUB -q gr10260f
#QSUB -W 24:00
#QSUB -A p=20:t=1:c=1:m=3072M
#QSUB -rn
#QSUB -J mp-1315-gruneisen-00-disp-002
#QSUB -e err.log
#QSUB -o std.log

mpirun ~/vasp535mpi
sleep 60