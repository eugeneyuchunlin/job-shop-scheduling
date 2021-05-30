#include <bits/types/FILE.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <include/common.h>
#include <memory.h>
#include <private/population.h>
#include <stdlib.h>
#include <type_traits>
#include "include/chromosome_base.h"
#include "include/job_base.h"
#include "include/linked_list.h"
#include "include/machine_base.h"
#include "private/job.h"


void random(double *genes, int size)
{
    for (int i = 0; i < size; ++i) {
        genes[i] = (double) rand() / (double) RAND_MAX;
    }
}

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
                               chromosome_base_t *chromosomes,
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

__global__ void initializeMachines(machine_t **machines,
                                   const int AMOUNT_OF_MACHINES,
                                   const int AMOUNT_OF_CHROMOSOMES)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < AMOUNT_OF_CHROMOSOMES && y < AMOUNT_OF_MACHINES) {
        initMachine(&machines[x][y]);
    }
}

__global__ void initializeChromosomes(chromosome_base_t *chromosomes,
                                      double **genes,
                                      const int AMOUNT_OF_JOBS,
                                      const int AMOUNT_OF_CHROMOSOMES)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < AMOUNT_OF_CHROMOSOMES) {
        chromosomes[x].chromosome_no = x;
        chromosomes[x].fitnessValue = 0;
        chromosomes[x].gene_size = AMOUNT_OF_JOBS << 1;
        chromosome_base_init(chromosomes + x, genes[x]);
    }
}

__global__ void machineSelection(job_t **jobs,
                                 job_base_operations_t *jbops,
                                 const int AMOUNT_OF_JOBS,
                                 const int AMOUNT_OF_CHROMOSOMES)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int machine_idx;
    if (x < AMOUNT_OF_CHROMOSOMES && y < AMOUNT_OF_JOBS) {
        machine_idx = jbops->machine_selection(&jobs[x][y].base);
        jobs[x][y].base.machine_no =
            jobs[x][y].base.process_time[machine_idx].machine_no;
        jobs[x][y].base.ptime =
            jobs[x][y].base.process_time[machine_idx].process_time;
    }
}

__global__ void machineSelection2(job_t **jobs,
                                  machine_t **machines,
                                  machine_base_operations_t *ops,
                                  const int AMOUNT_OF_JOBS,
                                  const int AMOUNT_OF_MACHINES,
                                  int AMOUNT_OF_CHROMOSOMES)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < AMOUNT_OF_CHROMOSOMES && y < AMOUNT_OF_MACHINES) {
        for (int i = 0; i < AMOUNT_OF_JOBS; ++i) {
            if (jobs[x][i].base.machine_no == machines[x][y].base.machine_no) {
                ops->add_job(&machines[x][y].base, &jobs[x][i].list);
            }
        }
    }
}

__global__ void sortJob(machine_t **machines,
                        list_operations_t *list_ops,
                        machine_base_operations_t *mbops,
                        const int AMOUNT_OF_MACHINES,
                        const int AMOUNT_OF_CHROMOSOMES)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < AMOUNT_OF_CHROMOSOMES && y < AMOUNT_OF_MACHINES) {
        mbops->sort_job(&machines[x][y].base, list_ops);
    }
}

__global__ void scheduling(machine_t **machines,
                           job_base_operations_t *jbops,
                           const int AMOUNT_OF_MACHINES,
                           const int AMOUNT_OF_CHROMOSOMES)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    list_ele_t *iter;
    job_t *job;
    job_t *prev = NULL;
    double start_time, arrival_time;

    if (x < AMOUNT_OF_CHROMOSOMES && y < AMOUNT_OF_MACHINES) {
        iter = machines[x][y].base.root;
        start_time = machines[x][y].base.avaliable_time;
        while (iter) {
            job = (job_t *) iter->ptr_derived_object;
            arrival_time = jbops->get_arrival_time(&job->base);
            if (prev != NULL && prev->recipe.ul_recipe !=
                                    job->recipe.ul_recipe) {  // setup time
                start_time += 20;
            }
            start_time =
                (start_time > arrival_time) ? start_time : arrival_time;

            jbops->set_start_time(&job->base, start_time);
            start_time = jbops->get_end_time(&job->base);
            iter = iter->next;
            prev = job;
        }
        machines[x][y].makespan = start_time;
    }
}

__global__ void computeFitnessValue(machine_t **machines,
                                    chromosome_base_t *chromosomes,
                                    const int AMOUNT_OF_MACHINES,
                                    const int AMOUNT_OF_CHROMOSOMES)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    double maxmakespan = -1;
    if (x < AMOUNT_OF_CHROMOSOMES) {
        for (int i = 0; i < AMOUNT_OF_MACHINES; ++i) {
            if (machines[x][i].makespan > maxmakespan) {
                maxmakespan = machines[x][i].makespan;
            }
        }
        chromosomes[x].fitnessValue = maxmakespan;
    }
}

__global__ void sortChromosomes(chromosome_base_t *chromosomes,
                                const int AMOUNT_OF_CHROMOSOMES)
{
    chromosome_base_t temp;
    for (int i = 0; i < AMOUNT_OF_CHROMOSOMES; ++i) {
        for (int j = i + 1; j < AMOUNT_OF_CHROMOSOMES; ++j) {
            if (chromosomes[i].fitnessValue > chromosomes[j].fitnessValue) {
                temp = chromosomes[i];
                chromosomes[i] = chromosomes[j];
                chromosomes[j] = temp;
            }
        }
    }
}

__global__ void getMachineJobs(machine_t **machines,
                               unsigned int *job_numbers,
                               double *seq,
                               unsigned int *size,
                               int CIDX,
                               int MIDX)
{
    list_ele_t *iter;
    iter = machines[CIDX][MIDX].base.root;
    int i = 0;
    job_t *job;
    while (iter) {
        job = ((job_t *) (iter->ptr_derived_object));
        job_numbers[i] = job->base.job_no;
        seq[i] = job->base.ptime;
        iter = iter->next;
        ++i;
    }
    *size = machines[CIDX][MIDX].base.size_of_jobs;
}



__global__ void resetMachines(machine_t **machines,
                              machine_base_operations_t *ops,
                              int AMOUNT_OF_MACHINES,
                              int AMOUNT_OF_CHROMOSOMES)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < AMOUNT_OF_CHROMOSOMES && y < AMOUNT_OF_MACHINES) {
        ops->reset(&machines[x][y].base);
        machines[x][y].makespan = 0;
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

    cudaCheck(
        cudaMalloc(
            (void **) &(pop->chromosomes.device_chromosome.chromosomes),
            sizeof(chromosome_base_t) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
        "cudaMalloc for chromosomes");
    double *genes;
    cudaCheck(
        cudaMallocHost((void **) &genes,
                       sizeof(double) * pop->parameters.AMOUNT_OF_JOBS << 1),
        "cudaMallocHost for genes in host");
    cudaCheck(
        cudaMalloc((void **) &(pop->chromosomes.device_chromosome.genes),
                   sizeof(double *) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
        "cudaMalloc for cu_genes");
    cudaCheck(
        cudaMallocHost(
            (void **) &(pop->chromosomes.device_chromosome.address_of_cugenes),
            sizeof(double *) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
        "cudaMallocHost for address_of_cugenes");
    for (int i = 0; i < pop->parameters.AMOUNT_OF_CHROMOSOMES; ++i) {
        random(genes, pop->parameters.AMOUNT_OF_JOBS << 1);
        cudaCheck(
            cudaMalloc(
                (void **) &(
                    pop->chromosomes.device_chromosome.address_of_cugenes[i]),
                sizeof(double) * pop->parameters.AMOUNT_OF_JOBS << 1),
            "cudaMalloc for genes for a chromosomes");
        cudaCheck(
            cudaMemcpy(pop->chromosomes.device_chromosome.address_of_cugenes[i],
                       genes,
                       sizeof(double) * pop->parameters.AMOUNT_OF_JOBS << 1,
                       cudaMemcpyHostToDevice),
            "cudaMemcpy genes from host to device");
    }

    cudaCheck(
        cudaMemcpy(pop->chromosomes.device_chromosome.genes,
                   pop->chromosomes.device_chromosome.address_of_cugenes,
                   sizeof(double *) * pop->parameters.AMOUNT_OF_CHROMOSOMES,
                   cudaMemcpyHostToDevice),
        "memcpy genes from host to device");
    cudaFreeHost(genes);

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
            cudaMalloc((void **) &(
                           pop->host_objects.address_of_process_times_entry[i]),
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


    //****************************Initialization************************//

    dim3 machine_chromosome_thread(100, 10);
    dim3 machine_chromosome_block(pop->parameters.AMOUNT_OF_CHROMOSOMES / 100,
                                  1);

    dim3 job_chromosome_thread(10, 100);
    dim3 job_chromosome_block(pop->parameters.AMOUNT_OF_CHROMOSOMES / 10, 1);


    initializeOps<<<1, 1>>>(pop->cuda_objects.list_ops,
                            pop->cuda_objects.job_ops,
                            pop->cuda_objects.machine_ops);

    initializeChromosomes<<<1, pop->parameters.AMOUNT_OF_CHROMOSOMES>>>(
        pop->chromosomes.device_chromosome.chromosomes,
        pop->chromosomes.device_chromosome.genes,
        pop->parameters.AMOUNT_OF_JOBS, pop->parameters.AMOUNT_OF_CHROMOSOMES);

    initializeJobs<<<job_chromosome_block, job_chromosome_thread>>>(
        pop->cuda_objects.jobs, pop->cuda_objects.process_times,
        pop->cuda_objects.job_ops,
        pop->chromosomes.device_chromosome.chromosomes,
        pop->parameters.AMOUNT_OF_JOBS, pop->parameters.AMOUNT_OF_CHROMOSOMES);


    initializeMachines<<<machine_chromosome_block, machine_chromosome_thread>>>(
        pop->cuda_objects.machines, pop->parameters.AMOUNT_OF_MACHINES,
        pop->parameters.AMOUNT_OF_CHROMOSOMES);
}


void geneticAlgorithm(struct population_t *pop)
{
    job_t *jobs;
    machine_t *machines;
    chromosome_base_t *chromosomes;

    cudaCheck(cudaMallocHost((void **) &jobs,
                             sizeof(job_t) * pop->parameters.AMOUNT_OF_JOBS),
              "cudaMallocHost for jobs in algorithm");
    cudaCheck(
        cudaMallocHost((void **) &machines,
                       sizeof(machine_t) * pop->parameters.AMOUNT_OF_MACHINES),
        "cudaMallocHost for machines in algorithm");

    dim3 machine_chromosome_thread(100, 10);
    dim3 machine_chromosome_block(pop->parameters.AMOUNT_OF_CHROMOSOMES / 100,
                                  1);

    dim3 job_chromosome_thread(10, 100);
    dim3 job_chromosome_block(pop->parameters.AMOUNT_OF_CHROMOSOMES / 10, 1);

    machineSelection<<<job_chromosome_block, job_chromosome_thread>>>(
        pop->cuda_objects.jobs, pop->cuda_objects.job_ops,
        pop->parameters.AMOUNT_OF_JOBS, pop->parameters.AMOUNT_OF_CHROMOSOMES);

    machineSelection2<<<machine_chromosome_block, machine_chromosome_thread>>>(
        pop->cuda_objects.jobs, pop->cuda_objects.machines,
        pop->cuda_objects.machine_ops, pop->parameters.AMOUNT_OF_JOBS,
        pop->parameters.AMOUNT_OF_MACHINES,
        pop->parameters.AMOUNT_OF_CHROMOSOMES);
    cudaDeviceSynchronize();
    sortJob<<<machine_chromosome_block, machine_chromosome_thread>>>(
        pop->cuda_objects.machines, pop->cuda_objects.list_ops,
        pop->cuda_objects.machine_ops, pop->parameters.AMOUNT_OF_MACHINES,
        pop->parameters.AMOUNT_OF_CHROMOSOMES);
    scheduling<<<machine_chromosome_block, machine_chromosome_thread>>>(
        pop->cuda_objects.machines, pop->cuda_objects.job_ops,
        pop->parameters.AMOUNT_OF_MACHINES,
        pop->parameters.AMOUNT_OF_CHROMOSOMES);
    computeFitnessValue<<<pop->parameters.AMOUNT_OF_CHROMOSOMES, 1>>>(
        pop->cuda_objects.machines,
        pop->chromosomes.device_chromosome.chromosomes,
        pop->parameters.AMOUNT_OF_MACHINES,
        pop->parameters.AMOUNT_OF_CHROMOSOMES);
    sortChromosomes<<<1, 1>>>(pop->chromosomes.device_chromosome.chromosomes,
                              pop->parameters.AMOUNT_OF_CHROMOSOMES);

    {
        cudaCheck(cudaMallocHost((void **) &chromosomes,
                                 sizeof(chromosome_base_t) *
                                     pop->parameters.AMOUNT_OF_CHROMOSOMES),
                  "cudaMalloc for chromosomes");
        cudaCheck(cudaMemcpy(chromosomes,
                             pop->chromosomes.device_chromosome.chromosomes,
                             sizeof(chromosome_base_t) *
                                 pop->parameters.AMOUNT_OF_CHROMOSOMES,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy chromosomes from device to host");
        for (int i = 0; i < pop->parameters.AMOUNT_OF_CHROMOSOMES; ++i) {
            printf("[%d]fitness value = %.3f\n", i,
                   chromosomes[i].fitnessValue);
        }
    }

    // test
    // {
    //     cudaCheck(cudaMemcpy(machines,
    //     pop->host_objects.address_of_cumachines[99],
    //     sizeof(machine_t)*pop->parameters.AMOUNT_OF_MACHINES,
    //     cudaMemcpyDeviceToHost), "cudaMemcpy machines99 from device to
    //     host"); printf("Happy Debugging\n");

    //     unsigned int * d_size, size;
    //     cudaCheck(cudaMalloc((void**)&d_size, sizeof(unsigned int)),
    //     "cudaMalloc for size"); unsigned int *d_job_numbers, *job_numbers;
    //     double * d_seq, *seq;

    //     cudaCheck(cudaMalloc((void**)&d_job_numbers, sizeof(unsigned
    //     int)*100), "cudaMalloc for d_job_numbers");
    //     cudaCheck(cudaMallocHost((void**)&job_numbers, sizeof(unsigned
    //     int)*100), "cudaMallocHost for job_numbers");
    //     cudaCheck(cudaMalloc((void**)&d_seq, sizeof(double)*100), "cudaMalloc
    //     for d_seq"); cudaCheck(cudaMallocHost((void**)&seq,
    //     sizeof(double)*100), "cudaMallocHost for seq");
    //
    //     getMachineJobs<<<1, 1>>>(pop->cuda_objects.machines, d_job_numbers,
    //     d_seq, d_size, 99, 0); cudaCheck(cudaMemcpy(&size, d_size,
    //     sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy size from
    //     device to host"); cudaCheck(cudaMemcpy(job_numbers, d_job_numbers,
    //     sizeof(unsigned int)*100, cudaMemcpyDeviceToHost), "cudaMemcpy
    //     job_number from device to host"); cudaCheck(cudaMemcpy(seq, d_seq,
    //     sizeof(double)*100, cudaMemcpyDeviceToHost), "cudaMemcpy seq from
    //     device to host"); printf("size is %u\n", size); for(unsigned int i =
    //     0; i < size; ++i){
    //         printf("%-10d", job_numbers[i]);
    //     }
    //     printf("\n");
    //     for(unsigned int i = 0; i < size; ++i){
    //         printf("%-9.2f ", seq[i]);
    //     }

    //     printf("\n");
    // }
}
