#include "../include/chromosome_base.h"
#include <include/chromosome_base.h>
#include <stdlib.h>

chromosome_base_t *chromosome_base_new(size_t gene_size, int chromosome_no)
{
    chromosome_base_t *chromosome =
        (chromosome_base_t *) malloc(sizeof(chromosome_base_t));

    if (!chromosome)
        return NULL;

    chromosome_base_init(chromosome, NULL);
    chromosome->gene_size = gene_size;
    chromosome->chromosome_no = chromosome_no;
    return chromosome;
}
__qualifier__ void chromosome_base_reset(chromosome_base_t *base)
{
    base->fitnessValue = 0;
}

__qualifier__ void chromosome_base_init(chromosome_base_t *base,
                                        double *address)
{
    int mid = base->gene_size >> 1;
    if (address) {
        base->genes = address;
    }
    base->ms_genes = base->genes;
    base->os_genes = base->ms_genes + mid;
    chromosome_base_reset(base);
}
