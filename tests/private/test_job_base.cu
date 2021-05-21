#include "include/job_base.h"
#include <tests/include/test_job_base.h>

job_t * newJob(int sizeof_pt){
	job_t * j = (job_t*)malloc(sizeof(job_t));
	job_base_init(&j->base);
	job_base_operations_t jbops = JOB_BASE_OPS;
	jbops.set_process_time(&j->base, NULL, sizeof_pt);
	return j;

}
