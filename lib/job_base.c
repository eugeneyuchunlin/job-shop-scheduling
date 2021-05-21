#include <include/job_base.h>
#include "include/linked_list.h"

// constructor and initialization

__qualifier__ void job_base_reset(job_base_t *self)
{
    self->start_time = 0;
    self->end_time = 0;
}


// setter
__qualifier__ void set_ms_gene_addr(job_base_t *self, double *ms_gene)
{
    self->ms_gene = ms_gene;
}

__qualifier__ void set_os_gene_addr(job_base_t *self, double *os_seq_gene)
{
    self->os_seq_gene = os_seq_gene;
}

__qualifier__ void set_process_time(job_base_t *self,
                                    process_time_t *ptime,
                                    unsigned int size_of_process_time)
{
    self->process_time = ptime;

    if (size_of_process_time != 0) {
        self->size_of_process_time = size_of_process_time;
        self->partition = 1.0 / (double) size_of_process_time;
    }
}

__qualifier__ void set_arrival_time(job_base_t *self, double arriv_time)
{
    self->arriv_t = arriv_time;
}

__qualifier__ void set_start_time(job_base_t *self, double start_time)
{
    self->start_time = start_time;
    if (self->process_time) {
        self->end_time = self->start_time +
                         self->process_time[self->machine_no].process_time;
    }
}

// getter
__qualifier__ double get_ms_gene(job_base_t *self)
{
    return *(self->ms_gene);
}

__qualifier__ double get_os_gene(job_base_t *self)
{
    return *(self->os_seq_gene);
}

__qualifier__ unsigned int get_machine_no(job_base_t *self)
{
    return self->machine_no;
}

__qualifier__ double get_arrival_time(job_base_t *self)
{
    return self->arriv_t;
}

__qualifier__ double get_start_time(job_base_t *self)
{
    return self->start_time;
}

__qualifier__ double get_end_time(job_base_t *self)
{
    return self->end_time;
}

// operation
__qualifier__ unsigned int machine_selection(job_base_t *self)
{
    // calculate which number of machine(from 1 to n) that corresponds to
    // partition

    double ms_gene = get_ms_gene(self);
    unsigned int val = ms_gene / self->partition + 0.0001;  // 0.0001 is bias
    if (self->process_time) {
        self->machine_no = self->process_time[val].machine_no;
    }
    return val;
}

__qualifier__ void job_base_init(void *_self)
{
    job_base_t *self = (job_base_t *) _self;
    self->ms_gene = self->os_seq_gene = NULL;
    self->process_time = NULL;
}


job_base_t *job_base_new()
{
    job_base_t *jb = (job_base_t *) malloc(sizeof(job_base_t));

    job_base_init(jb);
    return jb;
}
