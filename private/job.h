#ifndef __JOB_H__
#define __JOB_H__

#include <include/job_base.h>
#include <include/linked_list.h>

#include <map>
#include <string>
#include <set>

struct job_t{
    job_base_t base; 
    list_ele_t list;
    union{
        char str_recipe[8];
        unsigned long ul_recipe;
    }recipe;
    double urgent;
    double r_qt;
};

std::set<std::string> createJob(job_t * job, std::map<std::string, std::string> elements);

__device__ __host__ void initJob(job_t *);


__device__ __host__ double jobGetValue(void * _self);


#endif
