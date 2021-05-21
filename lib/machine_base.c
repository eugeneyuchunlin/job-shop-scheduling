#include <include/machine_base.h>
#include "include/job_base.h"
#include "include/linked_list.h"


__qualifier__ void machine_base_reset(machine_base_t *self)
{
    self->size_of_jobs = 0;
    self->root = self->tail = NULL;
}


__qualifier__ void _machine_base_add_job(machine_base_t *self, list_ele_t *job)
{
    job->next = job->prev = NULL;
    list_operations_t ops = LINKED_LIST_OPS;
    if (self->size_of_jobs == 0) {
        self->tail = self->root = job;
    } else {
        ops.set_next(self->tail, job);
        // self->tail->set_next(self->tail, job); // add into the list
        self->tail = job;  // move the tail
    }
    ++self->size_of_jobs;
}

__qualifier__ void _machine_base_sort_job(machine_base_t *self,
                                          list_operations_t *ops)
{
    if (self->size_of_jobs == 0) {
        return;
    }
    list_ele_t *ele = NULL;
    self->root = list_merge_sort(self->root, ops);
    ele = self->root;
    while (ele && ele->next) {
        ele = (list_ele_t *) ele->next;
    }
    self->tail = ele;
}


__qualifier__ unsigned int machine_base_get_size_of_jobs(machine_base_t *self)
{
    return self->size_of_jobs;
}


__qualifier__ void machine_base_init(machine_base_t *self)
{
    machine_base_reset(self);
}

machine_base_t *machine_base_new(unsigned int machine_no)
{
    machine_base_t *mb = (machine_base_t *) malloc(sizeof(machine_base_t));
    mb->machine_no = machine_no;
    return mb;
}
