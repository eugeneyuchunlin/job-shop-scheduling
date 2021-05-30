#include <bits/types/FILE.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <include/common.h>
#include <memory.h>
#include <private/population.h>
#include <stdlib.h>
#include "include/chromosome_base.h"
#include "include/job_base.h"
#include "include/linked_list.h"
#include "include/machine_base.h"

__global__ void initializeOps(list_operations_t *list_ops,
                              job_base_operations_t *job_ops,
                              machine_base_operations_t *machine_ops)
{
    *list_ops = LINKED_LIST_OPS;
    *job_ops = JOB_BASE_OPS;
    *machine_ops = MACHINE_BASE_OPS;
}

__global__ void initializeJobs(job_t **jobs,
                               process_time_t **process_times,
                               job_base_operations_t *ops,
                               chromosome_base_t * chromosomes,
                               const int AMOUNT_OF_JOBS,
                               const int AMOUNT_OF_CHROMOSOMES)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < AMOUNT_OF_CHROMOSOMES && y < AMOUNT_OF_JOBS) {
        initJob(&jobs[x][y]);
        ops->set_process_time(&jobs[x][y].base, process_times[y], 10);
        ops->set_ms_gene_addr(&jobs[x][y].base, chromosomes[x].ms_genes + y);
        ops->set_os_gene_addr(&jobs[x][y].base, chromosomes[x].os_genes + y);
    }
}

__global__ void initializeMachines(machine_t **machines, const int AMOUNT_OF_MACHINES, const int AMOUNT_OF_CHROMOSOMES){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x < AMOUNT_OF_CHROMOSOMES && y < AMOUNT_OF_MACHINES){
        initMachine(&machines[x][y]);     
    }
}

__global__ void initializeChromosomes(chromosome_base_t *chromosomes, double ** genes, const int AMOUNT_OF_JOBS, const int AMOUNT_OF_CHROMOSOMES){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < AMOUNT_OF_CHROMOSOMES){
        chromosomes[x].chromosome_no = x;
        chromosomes[x].fitnessValue = 0;
        chromosomes[x].gene_size = AMOUNT_OF_JOBS << 1;
        chromosome_base_init(chromosomes + x, genes[x]);
    }
}

void initPopulation(struct population_t *pop)
{
    job_t *jobs;
    machine_t *machines;
    process_time_t **process_times;
    // allocates page-locked memory for jobs
    cudaCheck(cudaMallocHost((void **) &jobs,
                             sizeof(job_t) * pop->parameters.AMOUNT_OF_JOBS),
              "cudaMallocHost for jobs");

    // copy jobs to page-locked memory
    memcpy(jobs, pop->sample.jobs,
           sizeof(job_t) * pop->parameters.AMOUNT_OF_JOBS);

    // allocates page-locked memory for machines
    cudaCheck(
        cudaMallocHost((void **) &machines,
                       sizeof(machine_t) * pop->parameters.AMOUNT_OF_MACHINES),
        "cudaMallocHost for machines");

    // copy machines to page-locked memory
    memcpy(machines, pop->sample.machines,
           sizeof(machine_t) * pop->parameters.AMOUNT_OF_MACHINES);

    // allocates page-locked memory for process_times
    cudaCheck(cudaMallocHost(
                  (void **) &process_times,
                  sizeof(process_time_t *) * pop->parameters.AMOUNT_OF_JOBS),
              "cudaMallocHost for process_times");

    for (int i = 0; i < pop->parameters.AMOUNT_OF_JOBS; ++i) {
        cudaCheck(cudaMallocHost((void **) &process_times[i],
                                 sizeof(process_time_t) * 10),
                  "cudaMallocHost for entry of process_times");
        memcpy(process_times[i], pop->sample.process_times[i],
               sizeof(process_time_t) * 10);
    }

    //==================Prepare chromosomes==================================//

    cudaCheck(cudaMalloc((void **) &(pop->chromosomes.device_chromosome.chromosomes),
                         sizeof(chromosome_base_t) *
                             pop->parameters.AMOUNT_OF_CHROMOSOMES),
              "cudaMalloc for chromosomes");

    cudaCheck(cudaMalloc((void**)&(pop->chromosomes.device_chromosome.genes), sizeof(double*)*pop->parameters.AMOUNT_OF_CHROMOSOMES), "cudaMalloc for cu_genes");
    cudaCheck(cudaMallocHost((void**)&(pop->chromosomes.device_chromosome.address_of_cugenes), sizeof(double*)*pop->parameters.AMOUNT_OF_CHROMOSOMES), "cudaMallocHost for address_of_cugenes");
    for(int i = 0; i < pop->parameters.AMOUNT_OF_CHROMOSOMES; ++i){
        cudaCheck(cudaMalloc((void**)&(pop->chromosomes.device_chromosome.address_of_cugenes[i]), sizeof(double)*pop->parameters.AMOUNT_OF_JOBS<<1), "cudaMalloc for genes for a chromosomes");
    }
    cudaCheck(cudaMemcpy(pop->chromosomes.device_chromosome.genes, pop->chromosomes.device_chromosome.address_of_cugenes, sizeof(double*)*pop->parameters.AMOUNT_OF_CHROMOSOMES, cudaMemcpyHostToDevice), "memcpy genes from host to device");

    //==================Prepare Jobs==========================================//
    cudaCheck(
        cudaMallocHost((void **) &(pop->host_objects.address_of_cujobs),
                       sizeof(job_t *) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
        "cudaMallocHost for address_of_cujobs");
    for (int i = 0; i < pop->parameters.AMOUNT_OF_CHROMOSOMES; ++i) {
        cudaCheck(
            cudaMalloc((void **) &(pop->host_objects.address_of_cujobs[i]),
                       sizeof(job_t) * pop->parameters.AMOUNT_OF_JOBS),
            "cudaMalloc for a bunch of jobs");
        cudaCheck(cudaMemcpy(pop->host_objects.address_of_cujobs[i], jobs,
                             sizeof(job_t) * pop->parameters.AMOUNT_OF_JOBS,
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy for jobs");
    }
    cudaCheck(
        cudaMalloc((void **) &(pop->cuda_objects.jobs),
                   sizeof(job_t *) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
        "cudaMalloc for cuda_jobs");
    cudaCheck(
        cudaMemcpy(pop->cuda_objects.jobs, pop->host_objects.address_of_cujobs,
                   sizeof(job_t *) * pop->parameters.AMOUNT_OF_CHROMOSOMES,
                   cudaMemcpyHostToDevice),
        "cudaMemcpy from host_objects.address_of_cujobs to "
        "pop->cuda_objects.jobs");

    //============Prepare Mahcines===========================================//

    cudaCheck(cudaMallocHost(
                  (void **) &(pop->host_objects.address_of_cumachines),
                  sizeof(machine_t *) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
              "cudaMallocHost for machines");

    for (int i = 0; i < pop->parameters.AMOUNT_OF_CHROMOSOMES; ++i) {
        cudaCheck(
            cudaMalloc((void **) &pop->host_objects.address_of_cumachines[i],
                       sizeof(machine_t) * pop->parameters.AMOUNT_OF_MACHINES),
            "cudaMalloc for bunch of machines");
        cudaCheck(
            cudaMemcpy(pop->host_objects.address_of_cumachines[i], machines,
                       sizeof(machine_t) * pop->parameters.AMOUNT_OF_MACHINES,
                       cudaMemcpyHostToDevice),
            "cudaMemcpy for machines");
    }
    cudaCheck(
        cudaMalloc((void **) &(pop->cuda_objects.machines),
                   sizeof(machine_t *) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
        "cudaMalloc for cuda_machines");

    cudaCheck(
        cudaMemcpy(pop->cuda_objects.machines,
                   pop->host_objects.address_of_cumachines,
                   sizeof(machine_t *) * pop->parameters.AMOUNT_OF_CHROMOSOMES,
                   cudaMemcpyHostToDevice),
        "cudaMemcpy from host_objects.address_of_cumachines to "
        "pop->cuda_objects.machines");

    //===========Prepare Process times ======================================//

    cudaCheck(
        cudaMalloc((void **) &(pop->cuda_objects.process_times),
                   sizeof(process_time_t *) * pop->parameters.AMOUNT_OF_JOBS),
        "cudaMalloc for process_times");

    cudaCheck(cudaMallocHost(
                  (void **) &(pop->host_objects.address_of_process_times_entry),
                  sizeof(process_time_t *) * pop->parameters.AMOUNT_OF_JOBS),
              "cudaMallocHost for address of process_times_entry");

    for (int i = 0; i < pop->parameters.AMOUNT_OF_JOBS; ++i) {
        cudaCheck(
            cudaMalloc((void **) &(pop->host_objects.address_of_process_times_entry[i]),
                       sizeof(process_time_t) * 10),
            "cudaMalloc for entry of process_times");
        cudaCheck(
            cudaMemcpy(pop->host_objects.address_of_process_times_entry[i],
                       pop->sample.process_times[i],
                       sizeof(process_time_t) * 10, cudaMemcpyHostToDevice),
            "cudaMemcpy for entry of process_times");
    }


    cudaCheck(
        cudaMemcpy(pop->cuda_objects.process_times,
                   pop->host_objects.address_of_process_times_entry,
                   sizeof(process_time_t *) * pop->parameters.AMOUNT_OF_JOBS,
                   cudaMemcpyHostToDevice),
        "cudaMemcpy from host_objects.address_of_process_times_entry to "
        "cuda_objects.process_times");

    //=========Prepare operation objects=================================//

    cudaCheck(cudaMalloc((void **) &pop->cuda_objects.list_ops,
                         sizeof(list_operations_t)),
              "cudaMalloc for list_ops");
    cudaCheck(cudaMalloc((void **) &pop->cuda_objects.job_ops,
                         sizeof(job_base_operations_t)),
              "cudaMalloc for job_ops");
    cudaCheck(cudaMalloc((void **) &pop->cuda_objects.machine_ops,
                         sizeof(machine_base_operations_t)),
              "cudaMalloc for machine_ops");

    dim3 machine_chromosome_thread(100, 10);
    dim3 machine_chromosome_block(pop->parameters.AMOUNT_OF_CHROMOSOMES / 100,
                                  1);

    dim3 job_chromosome_thread(10, 100);
    dim3 job_chromosome_block(pop->parameters.AMOUNT_OF_CHROMOSOMES / 10,
                              1);


    //****************************Initialization************************//
    initializeOps<<<1, 1>>>(pop->cuda_objects.list_ops,
                            pop->cuda_objects.job_ops,
                            pop->cuda_objects.machine_ops);

    initializeChromosomes<<<1, pop->parameters.AMOUNT_OF_CHROMOSOMES>>>(pop->chromosomes.device_chromosome.chromosomes, pop->chromosomes.device_chromosome.genes, pop->parameters.AMOUNT_OF_JOBS, pop->parameters.AMOUNT_OF_CHROMOSOMES);

    initializeJobs<<<job_chromosome_block, job_chromosome_thread>>>(
        pop->cuda_objects.jobs, pop->cuda_objects.process_times,
        pop->cuda_objects.job_ops, pop->chromosomes.device_chromosome.chromosomes, pop->parameters.AMOUNT_OF_JOBS,
        pop->parameters.AMOUNT_OF_CHROMOSOMES);

    
    initializeMachines<<<machine_chromosome_block, machine_chromosome_thread>>>(pop->cuda_objects.machines, pop->parameters.AMOUNT_OF_MACHINES, pop->parameters.AMOUNT_OF_CHROMOSOMES);

}
