#include <include/machine_base.h>
#include <include/job_base.h>
#include <include/linked_list.h>
#include <tests/include/test_machine_base.h>


__device__ __host__ double jobGetValue(void *_self){
	list_ele_t * self = (list_ele_t *)_self;
	job_t * j = (job_t *)self->ptr_derived_object;
	return j->val;
}

__device__ __host__ void initJob(job_t *self){
	_list_init(&self->ele);
	self->ele.ptr_derived_object = self;
	self->ele.get_value = jobGetValue;

	job_base_init(&self->base);
	self->base.ptr_derived_object = self;
}

job_t * newJob(double val){
	job_t * j = (job_t *)malloc(sizeof(job_t));
	_list_init(&j->ele);
	job_base_init(&j->base);
	initJob(j);
	j->val = val;
	return j;
}

__device__ __host__ void initMachine(void *_self){
	Machine *self = (Machine*)_self;
	machine_base_init(&self->base);
}

Machine * newMachine(){
	Machine * m = (Machine*)malloc(sizeof(Machine));
	initMachine(m);
	return m;
}

