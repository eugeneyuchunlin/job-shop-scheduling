

#ifndef __CHROMOSOME_BASE_H__
#define __CHROMOSOME_BASE_H__

#include <include/def.h>
#include <include/job_base.h>
#include <include/machine_base.h>

typedef struct chromosome_base_t chromosome_base_t;
typedef struct chromosome_base_operations_t chromosome_base_operations_t;

struct chromosome_base_t {
    int chromosome_no;
    size_t gene_size;
    double *ms_genes;
    double *os_genes;
    double fitnessValue;
    double *genes;
};

struct chromosome_base_operations_t {
    void (*init)(void *self, double *address);
    void (*reset)(void *self);
    void (*compute_fitness_value)(void *self,
                                  machine_base_t *machines,
                                  unsigned int machine_sizes,
                                  machine_base_operations_t *op);
};


chromosome_base_t *chromosome_base_new(size_t gene_size);

__qualifier__ void chromosome_base_reset(chromosome_base_t *base);

__qualifier__ void chromosome_base_init(chromosome_base_t *base,
                                        double *address);


#endif
