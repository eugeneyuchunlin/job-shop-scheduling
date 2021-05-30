#include <private/machine.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "include/machine_base.h"

void createMachine(machine_t * machine, std::map<std::string, std::string> elements){
    machine->base.avaliable_time = std::stoi(elements["RECOVER_TIME"]);
    machine->base.machine_no = atoi(&elements["EQP_ID"].c_str()[3]);
}

__device__ __host__ void initMachine(machine_t *machine){
    machine_base_init(&machine->base);
    machine->base.ptr_derived_object = machine;
}
