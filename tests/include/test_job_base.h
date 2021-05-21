#ifndef __TEST_JOB_BASE_H__
#define __TEST_JOB_BASE_H__

#include <include/job_base.h>

typedef struct job_t Job;

struct job_t{
	job_base_t base;
};

job_t * newJob(int sizeof_pt);

#endif
