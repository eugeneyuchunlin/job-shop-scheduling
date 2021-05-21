#ifndef __TEST_MACHINE_BASE_H__
#define __TEST_MACHINE_BASE_H__

#include "include/linked_list.h"
#include <cstdlib>
#include <iostream>
#include <gtest/gtest.h>
#include <include/machine_base.h>
#include <include/job_base.h>
#include <include/common.h>

#include <include/def.h>

#ifdef MACHINE_BASE_OPS
#undef MACHINE_BASE_OPS
#endif

#define MACHINE_BASE_OPS machine_base_operations_t{\
	.init = initMachine, \
    .reset = machine_base_reset,\
	.add_job = _machine_base_add_job,\
	.sort_job = _machine_base_sort_job,\
	.get_size_of_jobs = machine_base_get_size_of_jobs\
}

struct job_t{
	job_base_t base;
	list_ele_t ele;
	double val;
};

struct Machine{
	unsigned int machine_no;
	machine_base_t base;
};

__qualifier__ double machineSortJobs(void * self);

__qualifier__ double jobGetValue(void *);

__qualifier__ void initMachine(void *self);

__qualifier__ void initJob(job_t * _self);

job_t * newJob(double val);

Machine *newMachine();


#endif
