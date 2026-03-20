#!/usr/bin/env bash

##############
# input args
##############
# arg #1 [required]  : job filename template (e.g. job_cea_he.sh.tmpl)
# arg #2 [required]  : input parameter filename template (e.g. test_blast_2d_replicated.ini.tmpl)
# arg #3 [optional]  : proj_id, project id (or project account) used to submit job
# arg #4 [optional]  : nb_bricks_x, number of AMR trees along x (default value: 1)
# arg #5 [optional]  : nb_bricks_y, number of AMR trees along y (default value: 1)
# arg #6 [optional]  : nb_bricks_z, number of AMR trees along z (default value: 1)

function puts_message() {
    echo "$1"
}

function usage() {
    puts_message "Usage: $0 [-j <string>] [-i <string> ] [-p <string>] [-g <integer> ] [-h <integer> ] [-k <integer> ]" 1>&2;
    puts_message "Example: ./submit_job_cea_he.sh -j ./weak_scaling_3d/job_cea_he.sh.tmpl -i ./weak_scaling_3d/test_blast_3d_replicated_cea_he.ini.tmpl -g 2 -h 2 -k 2" 1>&2;
    puts_message "This will submit a job using 8 MPI processes (1 GPU per MPI process if Kokkos Cuda backend enabled)." 1>&2;
    exit 1;
}

while getopts ":j:i:p:g:h:k:" o; do
    case "${o}" in
        j)
            job_filename=${OPTARG}
            ;;
        i)
            input_filename=${OPTARG}
            ;;
        p)
            proj_id=${OPTARG}
            ;;
        g)
            nb_brick_x=${OPTARG}
            ((nb_brick_x > 0)) || usage
            ;;
        h)
            nb_brick_y=${OPTARG}
            ((nb_brick_y > 0)) || usage
            ;;
        k)
            nb_brick_z=${OPTARG}
            ((nb_brick_z > 0)) || usage
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${job_filename}" ] ; then
    puts_message "You must specify a job filename !"
    usage
fi

if [ -z "${input_filename}" ] ; then
    puts_message "You must specify an input parameter filename !"
    usage
fi

# Project id is optional
# when no project id provided, deactivate the use of project id
if [ -z "${proj_id}" ] ; then
    proj_id_enable='#'
else
    proj_id_enable=''
fi

if [ -z "${nb_brick_x}" ] ; then
    nb_brick_x=1
fi

if [ -z "${nb_brick_y}" ] ; then
    nb_brick_y=1
fi

if [ -z "${nb_brick_z}" ] ; then
    nb_brick_z=1
fi

# 1 GPU per brick
nb_gpus=$((nb_brick_x*nb_brick_y*nb_brick_z))

# pre-process input template job filename and template input parameter file
job_filename_subst=${job_filename%.tmpl}
input_filename_subst=${input_filename/.ini.tmpl/_mpi_${nb_gpus}.ini}

# generate job and input file
NBRICK_X=${nb_brick_x} NBRICK_Y=${nb_brick_y} NBRICK_Z=${nb_brick_z} envsubst '${NBRICK_X},${NBRICK_Y},${NBRICK_Z}' < "${input_filename}" > "${input_filename_subst}"
PROJ_ID_ENABLE=${proj_id_enable} PROJECT_ID=${proj_id} NGPU=${nb_gpus} INI_FILENAME=${input_filename_subst} envsubst '${PROJECT_ID} ${NGPU} ${INI_FILENAME}' < ${job_filename} > ${job_filename_subst}

# submit job
job_submit=${JOB_SUBMIT:-ccc_msub}
${job_submit} "${job_filename_subst}"
