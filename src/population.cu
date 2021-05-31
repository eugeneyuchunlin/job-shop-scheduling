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
#include "private/job.h"


void random(double *genes, int size)
{
    for (int i = 0; i < size; ++i) {
        genes[i] = (double) rand() / (double) RAND_MAX;
    }
}

int random_range(int start, int end, int different_num)
{
    if (different_num < 0) {
        return start + rand() % (end - start);
    } else {
        int rnd = start + (rand() % (end - start));
        while (rnd == different_num) {
            rnd = start + (rand() % (end - start));
        }
        return rnd;
    }
}

void generateCrossoverFactors(struct evolution_factors_t *factors,
                              int factor_size,
                              int gene_size,
                              const int AMOUNT_OF_R_CHROMOSOMES)
{
    int tmp;
    for (int i = 0; i < factor_size; ++i) {
        tmp = factors->c_selected1[i] =
            random_range(0, AMOUNT_OF_R_CHROMOSOMES, -1);
        factors->c_selected2[i] = random_range(0, AMOUNT_OF_R_CHROMOSOMES, tmp);
        tmp = factors->cut_points[i] = random_range(0, gene_size, -1);
        factors->range[i] = random_range(0, gene_size - tmp, -1);
    }
}

void generateMutationFactors(struct evolution_factors_t *factors,
                             int factor_size,
                             int gene_size,
                             const int AMOUNT_OF_R_CHROMOSOMES)
{
    for (int i = 0; i < factor_size; ++i) {
        factors->m_selected[i] = random_range(0, AMOUNT_OF_R_CHROMOSOMES, -1);
        factors->gene_idx[i] = random_range(0, gene_size, -1);
    }
    random(factors->new_genes, factor_size);
}

void cpyEvolutionFactors(struct evolution_factors_t *dest,
                         struct evolution_factors_t *src,
                         const int AMOUNT_OF_R_CHROMOSOMES)
{
    cudaCheck(cudaMemcpy(dest->c_selected1, src->c_selected1,
                         sizeof(unsigned int) * AMOUNT_OF_R_CHROMOSOMES,
                         cudaMemcpyHostToDevice),
              "copy c_selected1");
    cudaCheck(cudaMemcpy(dest->c_selected2, src->c_selected2,
                         sizeof(unsigned int) * AMOUNT_OF_R_CHROMOSOMES,
                         cudaMemcpyHostToDevice),
              "copy c_selected2");
    cudaCheck(cudaMemcpy(dest->m_selected, src->m_selected,
                         sizeof(unsigned int) * AMOUNT_OF_R_CHROMOSOMES,
                         cudaMemcpyHostToDevice),
              "copy m_selected");
    cudaCheck(cudaMemcpy(dest->cut_points, src->cut_points,
                         sizeof(unsigned int) * AMOUNT_OF_R_CHROMOSOMES,
                         cudaMemcpyHostToDevice),
              "copy cut_points");
    cudaCheck(cudaMemcpy(dest->range, src->range,
                         sizeof(unsigned int) * AMOUNT_OF_R_CHROMOSOMES,
                         cudaMemcpyHostToDevice),
              "copy range");
    cudaCheck(cudaMemcpy(dest->new_genes, src->new_genes,
                         sizeof(double) * AMOUNT_OF_R_CHROMOSOMES,
                         cudaMemcpyHostToDevice),
              "copy new_genes");

    cudaCheck(cudaMemcpy(dest->gene_idx, src->gene_idx,
                         sizeof(unsigned int) * AMOUNT_OF_R_CHROMOSOMES,
                         cudaMemcpyHostToDevice),
              "copy gene_idx");
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

__global__ void binding(job_t **jobs,
                        chromosome_base_t *chromosomes,
                        job_base_operations_t *ops,
                        const int AMOUNT_OF_JOBS,
                        const int AMOUNT_OF_CHROMOSOMES)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < AMOUNT_OF_CHROMOSOMES && y < AMOUNT_OF_JOBS) {
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
    int scrapped = 0;

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
            if(start_time > job->r_qt){
                scrapped += 1; 
            }
            start_time = jbops->get_end_time(&job->base);
            
            iter = iter->next;
            prev = job;
        }
        machines[x][y].makespan = start_time;
        machines[x][y].scrapped = scrapped;
    }
}

__global__ void computeFitnessValue(machine_t **machines,
                                    chromosome_base_t *chromosomes,
                                    const int AMOUNT_OF_MACHINES,
                                    const int AMOUNT_OF_CHROMOSOMES)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    double maxmakespan = -1;
    int totalScrapped = 0;
    if (x < AMOUNT_OF_CHROMOSOMES) {
        for (int i = 0; i < AMOUNT_OF_MACHINES; ++i) {
            if (machines[x][i].makespan > maxmakespan) {
                maxmakespan = machines[x][i].makespan;
            }
            totalScrapped += machines[x][i].scrapped;
        }
        chromosomes[x].fitnessValue = maxmakespan + totalScrapped * 100;
    }
}

__global__ void sortChromosomes(chromosome_base_t *chromosomes,
                                const int AMOUNT_OF_CHROMOSOMES)
{
    chromosome_base_t temp;
    for (int i = 0; i < AMOUNT_OF_CHROMOSOMES - 1; ++i) {
        for (int j = 0; j < AMOUNT_OF_CHROMOSOMES - 1; ++j) {
            if (chromosomes[j].fitnessValue > chromosomes[j + 1].fitnessValue) {
                temp = chromosomes[j];
                chromosomes[j] = chromosomes[j + 1];
                chromosomes[j + 1] = temp;
            }
        }
    }
}


__global__ void crossover(chromosome_base_t *chromosomes,
                          unsigned int *selected1,
                          unsigned int *selected2,
                          unsigned int *cut_points,
                          unsigned int *range,
                          unsigned int offset,
                          const int AMOUNT_OF_JOBS,
                          const int AMOUNT_OF_FACTORS)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    double *gene1, *gene2, *r_gene1, *r_gene2;
    if (x < AMOUNT_OF_FACTORS) {
        gene1 = chromosomes[selected1[x]].genes;
        gene2 = chromosomes[selected2[x]].genes;
        r_gene1 = chromosomes[2 * x + offset + AMOUNT_OF_JOBS].genes;
        r_gene2 = chromosomes[2 * x + offset + AMOUNT_OF_JOBS + 1].genes;
        memcpy(r_gene1, gene1, sizeof(double) * AMOUNT_OF_JOBS * 2);
        memcpy(r_gene2, gene2, sizeof(double) * AMOUNT_OF_JOBS * 2);
        memcpy(r_gene1 + cut_points[x], gene2 + cut_points[x],
               sizeof(double) * range[x]);
        memcpy(r_gene2 + cut_points[x], gene1 + cut_points[x],
               sizeof(double) * range[x]);
    }
}


__global__ void mutation(chromosome_base_t *chromosomes,
                         unsigned int *selected,
                         unsigned int *gene_idx,
                         double *ngenes,
                         unsigned int offset,
                         const int AMOUNT_OF_JOBS,
                         const int AMOUNT_OF_MUTATIONS)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    double *gene, *r_gene;
    if (x < AMOUNT_OF_MUTATIONS) {
        gene = chromosomes[selected[x]].genes;
        r_gene = chromosomes[x + offset + AMOUNT_OF_JOBS].genes;
        memcpy(r_gene, gene, sizeof(double) * AMOUNT_OF_JOBS * 2);
        r_gene[gene_idx[x]] = ngenes[x];
    }
}

__global__ void getMachineJobs(machine_t **machines,
                               unsigned int *job_numbers,
                               double *start_time,
                               double *end_time,
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
        start_time[i] = job->base.start_time;
        end_time[i] = job->base.end_time;
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

    // host
    pop->chromosomes.host_chromosome.genes = (double **) malloc(
        sizeof(double *) *
        pop->chromosomes.host_chromosome.AMOUNT_OF_HOST_CHROMOSOMES);
    for (int i = 0;
         i < pop->chromosomes.host_chromosome.AMOUNT_OF_HOST_CHROMOSOMES; ++i) {
        cudaCheck(
            cudaMallocHost((void **) &pop->chromosomes.host_chromosome.genes[i],
                           sizeof(double) * pop->parameters.AMOUNT_OF_JOBS * 2),
            "cudaMallocHost for host genes");
    }
    cudaCheck(
        cudaMallocHost(
            (void **) &pop->chromosomes.host_chromosome.chromosomes,
            sizeof(chromosome_base_t) *
                pop->chromosomes.host_chromosome.AMOUNT_OF_HOST_CHROMOSOMES),
        "cudaMallocHost for host chromosomes");


    // device
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


    //=======Prepare evolution factors=================================//
    // device
    cudaCheck(cudaMalloc(
                  (void **) &pop->evolution_factors.device.c_selected1,
                  sizeof(unsigned int) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
              "cudaMalloc for c_selected1");

    cudaCheck(cudaMalloc(
                  (void **) &pop->evolution_factors.device.c_selected2,
                  sizeof(unsigned int) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
              "cudaMalloc for c_selected2");

    cudaCheck(cudaMalloc(
                  (void **) &pop->evolution_factors.device.cut_points,
                  sizeof(unsigned int) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
              "cudaMalloc for cut_points");

    cudaCheck(cudaMalloc(
                  (void **) &pop->evolution_factors.device.range,
                  sizeof(unsigned int) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
              "cudaMalloc for range");

    cudaCheck(cudaMalloc(
                  (void **) &pop->evolution_factors.device.m_selected,
                  sizeof(unsigned int) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
              "cudaMalloc for m_selected");

    cudaCheck(
        cudaMalloc((void **) &pop->evolution_factors.device.new_genes,
                   sizeof(double) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
        "cudaMalloc for m_selected");
    cudaCheck(cudaMalloc(
                  (void **) &pop->evolution_factors.device.gene_idx,
                  sizeof(unsigned int) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
              "cudaMalloc for gene_idx");



    // host
    cudaCheck(cudaMallocHost(
                  (void **) &pop->evolution_factors.host.c_selected1,
                  sizeof(unsigned int) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
              "cudaMallocHost for c_selected1");

    cudaCheck(cudaMallocHost(
                  (void **) &pop->evolution_factors.host.c_selected2,
                  sizeof(unsigned int) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
              "cudaMallocHost for c_selected2");

    cudaCheck(cudaMallocHost(
                  (void **) &pop->evolution_factors.host.cut_points,
                  sizeof(unsigned int) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
              "cudaMallocHost for cut_points");

    cudaCheck(cudaMallocHost(
                  (void **) &pop->evolution_factors.host.range,
                  sizeof(unsigned int) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
              "cudaMallocHost for range");

    cudaCheck(cudaMallocHost(
                  (void **) &pop->evolution_factors.host.m_selected,
                  sizeof(unsigned int) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
              "cudaMallocHost for m_selected");

    cudaCheck(
        cudaMallocHost((void **) &pop->evolution_factors.host.new_genes,
                       sizeof(double) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
        "cudaMallocHost for m_selected");

    cudaCheck(cudaMallocHost(
                  (void **) &pop->evolution_factors.host.gene_idx,
                  sizeof(unsigned int) * pop->parameters.AMOUNT_OF_CHROMOSOMES),
              "cudaMallocHost for gene_idx");



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
    cudaDeviceSynchronize();
}

void copyResult(struct population_t *pop, char *filename)
{
    {
        job_t *jobs;
        machine_t *machines;
        chromosome_base_t *chromosomes;
        FILE * file = fopen(filename, "w");

        cudaCheck(
            cudaMallocHost((void **) &jobs,
                           sizeof(job_t) * pop->parameters.AMOUNT_OF_JOBS),
            "cudaMallocHost for jobs in algorithm");
        cudaCheck(cudaMallocHost(
                      (void **) &machines,
                      sizeof(machine_t) * pop->parameters.AMOUNT_OF_MACHINES),
                  "cudaMallocHost for machines in algorithm");

        cudaCheck(cudaMallocHost((void **) &chromosomes,
                                 sizeof(chromosome_base_t) *
                                     pop->parameters.AMOUNT_OF_CHROMOSOMES),
                  "cudaMalloc for chromosomes");

        cudaCheck(
            cudaMemcpy(machines, pop->host_objects.address_of_cumachines[26],
                       sizeof(machine_t) * pop->parameters.AMOUNT_OF_MACHINES,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy machines26 from device to host");
        // printf("Happy Debugging\n");

        unsigned int *d_size, size;
        cudaCheck(cudaMalloc((void **) &d_size, sizeof(unsigned int)),
                  "cudaMalloc for size");
        unsigned int *d_job_numbers, *job_numbers;
        double *d_seq, *seq, *end_time, *d_end_time;

        cudaCheck(
            cudaMalloc((void **) &d_job_numbers, sizeof(unsigned int) * 100),
            "cudaMalloc for d_job_numbers");
        cudaCheck(
            cudaMallocHost((void **) &job_numbers, sizeof(unsigned int) * 100),
            "cudaMallocHost for job_numbers");
        cudaCheck(cudaMalloc((void **) &d_seq, sizeof(double) * 100),
                  "cudaMalloc"
                  "for d_seq");
        cudaCheck(cudaMallocHost((void **) &seq, sizeof(double) * 100),
                  "cudaMallocHost for seq");
       
        cudaCheck(cudaMalloc((void **) &d_end_time, sizeof(double) * 100),
                  "cudaMalloc"
                  "for d_end_time");
        cudaCheck(cudaMallocHost((void **) &end_time, sizeof(double) * 100),
                  "cudaMallocHost for end_time");
       


        for (unsigned int i = 0; i < 10; ++i) {
            getMachineJobs<<<1, 1>>>(pop->cuda_objects.machines, d_job_numbers,
                                     d_seq, d_end_time, d_size, 0, i);

            cudaCheck(cudaMemcpy(&size, d_size, sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy size from"
                      "device to host");

            cudaCheck(
                cudaMemcpy(job_numbers, d_job_numbers,
                           sizeof(unsigned int) * 100, cudaMemcpyDeviceToHost),
                "cudaMemcpy job_number from device to host");

            cudaCheck(cudaMemcpy(seq, d_seq, sizeof(double) * 100,
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy seq from device to host");

            cudaCheck(cudaMemcpy(end_time, d_end_time, sizeof(double) * 100,
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy seq from device to host");

            for(unsigned int j = 0; j < size; ++j){
                fprintf(file, "%d %d %.3f %.3f\n", job_numbers[j], i, seq[j], end_time[j]);
            }

        }
        fclose(file);
    }
    
}

void *geneticAlgorithm(void *_pop)
{
    struct population_t *pop = (struct population_t *) _pop;
    job_t *jobs;
    machine_t *machines;
    chromosome_base_t *chromosomes;
    int CROSSOVER_AMOUNT;
    int MUTATION_AMOUNT;
    int fitnessVaule, fitnessCounter;
    fitnessVaule = 0;
    fitnessCounter = 0;

    cudaCheck(cudaMallocHost((void **) &jobs,
                             sizeof(job_t) * pop->parameters.AMOUNT_OF_JOBS),
              "cudaMallocHost for jobs in algorithm");
    cudaCheck(
        cudaMallocHost((void **) &machines,
                       sizeof(machine_t) * pop->parameters.AMOUNT_OF_MACHINES),
        "cudaMallocHost for machines in algorithm");

    cudaCheck(cudaMallocHost((void **) &chromosomes,
                             sizeof(chromosome_base_t) *
                                 pop->parameters.AMOUNT_OF_CHROMOSOMES),
              "cudaMalloc for chromosomes");


    dim3 machine_chromosome_thread(100, 10);
    dim3 machine_chromosome_block(pop->parameters.AMOUNT_OF_CHROMOSOMES / 100,
                                  1);

    dim3 job_chromosome_thread(10, 100);
    dim3 job_chromosome_block(pop->parameters.AMOUNT_OF_CHROMOSOMES / 10, 1);

    for (int i = 0; i < pop->parameters.GENERATIONS; ++i) {
        binding<<<job_chromosome_block, job_chromosome_thread>>>(
            pop->cuda_objects.jobs,
            pop->chromosomes.device_chromosome.chromosomes,
            pop->cuda_objects.job_ops, pop->parameters.AMOUNT_OF_JOBS,
            pop->parameters.AMOUNT_OF_CHROMOSOMES);

        resetMachines<<<machine_chromosome_block, machine_chromosome_thread>>>(
            pop->cuda_objects.machines, pop->cuda_objects.machine_ops,
            pop->parameters.AMOUNT_OF_MACHINES,
            pop->parameters.AMOUNT_OF_CHROMOSOMES);


        machineSelection<<<job_chromosome_block, job_chromosome_thread>>>(
            pop->cuda_objects.jobs, pop->cuda_objects.job_ops,
            pop->parameters.AMOUNT_OF_JOBS,
            pop->parameters.AMOUNT_OF_CHROMOSOMES);

        machineSelection2<<<machine_chromosome_block,
                            machine_chromosome_thread>>>(
            pop->cuda_objects.jobs, pop->cuda_objects.machines,
            pop->cuda_objects.machine_ops, pop->parameters.AMOUNT_OF_JOBS,
            pop->parameters.AMOUNT_OF_MACHINES,
            pop->parameters.AMOUNT_OF_CHROMOSOMES);

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
        sortChromosomes<<<1, 1>>>(
            pop->chromosomes.device_chromosome.chromosomes,
            pop->parameters.AMOUNT_OF_CHROMOSOMES);

        CROSSOVER_AMOUNT = pop->parameters.AMOUNT_OF_R_CHROMOSOMES *
                           pop->parameters.EVOLUTION_RATE;

        MUTATION_AMOUNT =
            pop->parameters.AMOUNT_OF_R_CHROMOSOMES - CROSSOVER_AMOUNT;


        generateCrossoverFactors(&pop->evolution_factors.host,
                                 CROSSOVER_AMOUNT >> 1,
                                 pop->parameters.AMOUNT_OF_JOBS << 1,
                                 pop->parameters.AMOUNT_OF_R_CHROMOSOMES);

        generateMutationFactors(&pop->evolution_factors.host, MUTATION_AMOUNT,
                                pop->parameters.AMOUNT_OF_JOBS << 1,
                                pop->parameters.AMOUNT_OF_R_CHROMOSOMES);
        cpyEvolutionFactors(&pop->evolution_factors.device,
                            &pop->evolution_factors.host,
                            pop->parameters.AMOUNT_OF_R_CHROMOSOMES);

        cudaDeviceSynchronize();
        crossover<<<1, CROSSOVER_AMOUNT>>>(
            pop->chromosomes.device_chromosome.chromosomes,
            pop->evolution_factors.device.c_selected1,
            pop->evolution_factors.device.c_selected2,
            pop->evolution_factors.device.cut_points,
            pop->evolution_factors.device.range, 0,
            pop->parameters.AMOUNT_OF_JOBS, CROSSOVER_AMOUNT >> 1);

        mutation<<<1, MUTATION_AMOUNT>>>(
            pop->chromosomes.device_chromosome.chromosomes,
            pop->evolution_factors.device.m_selected,
            pop->evolution_factors.device.gene_idx,
            pop->evolution_factors.device.new_genes, CROSSOVER_AMOUNT,
            pop->parameters.AMOUNT_OF_JOBS, MUTATION_AMOUNT);


        {
            cudaCheck(cudaMemcpy(chromosomes,
                                 pop->chromosomes.device_chromosome.chromosomes,
                                 sizeof(chromosome_base_t) *
                                     pop->parameters.AMOUNT_OF_CHROMOSOMES,
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy chromosomes from device to host");
            printf("(%u-%d)[%d]fitness value = %.3f\n", pop->population_number,
                   i, chromosomes[0].chromosome_no,
                   chromosomes[0].fitnessValue);
            pop->best_fitness_value = chromosomes[0].fitnessValue;
            if (chromosomes[0].fitnessValue - fitnessVaule < 0.0001) {
                fitnessVaule = chromosomes[0].fitnessValue;
                fitnessCounter = 0;
            } else {
                ++fitnessCounter;
            }
            if (fitnessCounter > 50) {
                pop->parameters.EVOLUTION_RATE -= 0.05;
                if (pop->parameters.EVOLUTION_RATE < 0) {
                    break;
                }
                fitnessCounter = 0;
            }

            // printf("Counter = %d\n", fitnessCounter);
            // printf("=========================\n");
        }
    }
    cudaDeviceSynchronize();
    pthread_exit(NULL);
}

void swapPopulation(struct population_t pops[], const int amount_of_populations)
{
    // copy chromosomes from device to host
    for (int i = 0, size = amount_of_populations; i < size; ++i) {
        cudaCheck(
            cudaMemcpy(
                pops[i].chromosomes.host_chromosome.chromosomes,
                pops[i].chromosomes.device_chromosome.chromosomes,
                sizeof(chromosome_base_t) *
                    pops[i]
                        .chromosomes.host_chromosome.AMOUNT_OF_HOST_CHROMOSOMES,
                cudaMemcpyDeviceToHost),
            "copy chromosomes from device to host");
    }

    // copy genes
    for (int i = 0, size = amount_of_populations; i < size; ++i) {
        for (int j = 0, size_c = pops[i]
                                     .chromosomes.host_chromosome
                                     .AMOUNT_OF_HOST_CHROMOSOMES;
             j < size_c; ++j) {
            cudaCheck(
                cudaMemcpy(
                    pops[i].chromosomes.host_chromosome.genes[j],
                    pops[i].chromosomes.host_chromosome.chromosomes[j].genes,
                    sizeof(double) * pops[i].parameters.AMOUNT_OF_JOBS << 1,
                    cudaMemcpyDeviceToHost),
                "copy genes from device to host");
        }
    }

    // swap
    for (int i = 0, size = amount_of_populations - 1; i < size; ++i) {
        for (int j = 0, size_c = pops[i]
                                     .chromosomes.host_chromosome
                                     .AMOUNT_OF_HOST_CHROMOSOMES;
             j < size_c; ++j) {
            cudaCheck(
                cudaMemcpy(
                    pops[i + 1]
                        .chromosomes.host_chromosome.chromosomes[j]
                        .genes,
                    pops[i].chromosomes.host_chromosome.genes[j],
                    sizeof(double) * pops[i].parameters.AMOUNT_OF_JOBS << 1,
                    cudaMemcpyHostToDevice),
                "copy genes from host to device");
        }
    }

    for (int i = 0,
             size_c =
                 pops[i].chromosomes.host_chromosome.AMOUNT_OF_HOST_CHROMOSOMES;
         i < size_c; ++i) {
        cudaCheck(
            cudaMemcpy(pops[0].chromosomes.host_chromosome.chromosomes[i].genes,
                       pops[amount_of_populations - 1]
                           .chromosomes.host_chromosome.genes[i],
                       sizeof(double) * pops[0].parameters.AMOUNT_OF_JOBS << 1,
                       cudaMemcpyHostToDevice),
            "copy genes from host to device");
    }
}
