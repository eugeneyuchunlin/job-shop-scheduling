#ifndef __MACHINE_H__
#define __MACHINE_H__

#include <include/machine_base.h>
#include <map>
#include <string>

typedef struct machine_t machine_t;

struct machine_t{
    machine_base_t  base;
    double makespan;
    int scrapped;
};

void createMachine(machine_t * machine, std::map<std::string, std::string> elements);

__device__ __host__ void initMachine(machine_t *machine);
#endif
