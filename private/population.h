#ifndef __POPULATION_H__
#define __POPULATION_H__

#include "include/job_base.h"
#include "include/linked_list.h"
#include "include/chromosome_base.h"
#include "include/machine_base.h"
#include "job.h"
#include "machine.h"


struct evolution_factors_t{
    unsigned int *c_selected1;
    unsigned int *c_selected2;
    unsigned int *cut_points;
    unsigned int *range;

    unsigned int *m_selected;
    unsigned int *gene_idx;
    double *new_genes;
};

struct population_t{
    unsigned int population_number;

    struct {
        const int AMOUNT_OF_JOBS;
        const int AMOUNT_OF_MACHINES;
        const int AMOUNT_OF_CHROMOSOMES;
        const int AMOUNT_OF_R_CHROMOSOMES;
        double EVOLUTION_RATE;
        double SELECTION_RATE;
    }parameters;

    struct {
        job_t * jobs;
        machine_t * machines;
        process_time_t ** process_times;
    }sample;

    struct{
        job_t ** jobs;
        machine_t **machines;
        process_time_t ** process_times;
        list_operations_t *list_ops;
        job_base_operations_t *job_ops;
        machine_base_operations_t *machine_ops;
    }cuda_objects;

    struct{
        job_t ** address_of_cujobs;
        machine_t ** address_of_cumachines;
        process_time_t **address_of_process_times_entry;    
    }host_objects;

    struct {
        int AMOUNT_OF_HOST_CHROMOSOMES;
        struct{
            chromosome_base_t * chromosomes;
            double **genes;
        }host_chromosome;
        struct{
            chromosome_base_t * chromosomes;
            double **genes;
            double **address_of_cugenes;
        }device_chromosome;
    }chromosomes;

    struct {
        struct evolution_factors_t device;
        struct evolution_factors_t host;
    }evolution_factors;
};


void initPopulation(struct population_t * pop);

void geneticAlgorithm(struct population_t * pop);

#endif
