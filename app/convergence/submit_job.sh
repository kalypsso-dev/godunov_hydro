#!/usr/bin/env bash

##############
# input args
##############
# arg #1 [required]  : -j job filename template (e.g. job.sh.tmpl)
# arg #2 [required]  : -i input parameter filename template (e.g. test_isentropic_vortex_2d.ini.tmpl)
# arg #3 [required]  : -m nb_mpi_tasks, number of MPI tasks (default value: 1)
# arg #4 [required]  : -b block size
# arg #5 [required]  : -k level_min
# arg #6 [required]  : -l level_max
# arg #7 [optional]  : -p proj_id, project id (or project account) used to submit job

function puts_message() {
    echo "$1"
}

function usage() { puts_message "Usage: $0 [-j <string>] [-i <string>] [-p <string>] [-m <integer>] [-b <integer>] [-k <integer>] [-l <integer>]" 1>&2; exit 1; }

while getopts ":j:i:p:b:k:l:m:" o; do
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
        b)
            block_size=${OPTARG}
            ;;
        k)
            level_min=${OPTARG}
            ;;
        l)
            level_max=${OPTARG}
            ;;
        m)
            nb_mpi_tasks=${OPTARG}
            ((nb_mpi_tasks > 0)) || usage
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

if [ -z "${block_size}" ] ; then
    puts_message "You must specify a block size !"
    usage
fi

# Project id is optional
# when no project id provided, deactivate the use of project id
if [ -z "${proj_id}" ] ; then
    proj_id_enable='#'
else
    proj_id_enable=''
fi

# pre-process input template job filename and template input parameter file
job_filename_subst=${job_filename%.tmpl}
input_filename_subst=${input_filename/.ini.tmpl/_mpi_${nb_mpi_tasks}_bs_${block_size}_level_${level_min}_${level_max}.ini}

# generate job and input file
BLOCK_SIZE=${block_size} LEVEL_MIN=${level_min} LEVEL_MAX=${level_max} envsubst '${BLOCK_SIZE} ${LEVEL_MIN} ${LEVEL_MAX}'< ${input_filename} > ${input_filename_subst}
PROJ_ID_ENABLE=${proj_id_enable} PROJECT_ID=${proj_id} NB_MPI_TASKS=${nb_mpi_tasks} BLOCK_SIZE=${block_size} LEVEL_MIN=${level_min} LEVEL_MAX=${level_max} INI_FILENAME=${input_filename_subst} envsubst '${PROJECT_ID} ${NB_MPI_TASKS} ${INI_FILENAME} ${BLOCK_SIZE} ${LEVEL_MIN} ${LEVEL_MAX} ${PROJ_ID_ENABLE}' < ${job_filename} > ${job_filename_subst}

# submit job
job_submit=${JOB_SUBMIT:-ccc_msub}
${job_submit} ${job_filename_subst}
