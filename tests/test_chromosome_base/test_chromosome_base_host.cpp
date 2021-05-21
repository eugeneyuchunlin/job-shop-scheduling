#include "include/job_base.h"
#include "include/linked_list.h"
#include "include/machine_base.h"
#include <cstdlib>
#include <gtest/gtest.h>

#include <include/job_base.h>
#include <include/chromosome_base.h>
#include <tests/include/test_chromosome_base.h>
#include <tests/include/test_machine_base.h>

extern int JOB_AMOUNT;
extern int MACHINE_AMOUNT;
extern int CHROMOSOME_AMOUNT;
extern int GENERATIONS;


class TestChromosomeBaseHost :public testing::Test{
public:
	Machine * machines;
	job_t * jobs;
	Chromosome * chromosomes;
	void random_shuffle(chromosome_base_t *chromosome);
	void SetUp() override;
	void TearDown() override;
};

void TestChromosomeBaseHost::random_shuffle(chromosome_base_t * chromosome){
	for(unsigned int i = 0; i < chromosome->gene_size; ++i){
		chromosome->genes[i] = (double)rand() / (double)RAND_MAX;
	}
}

void TestChromosomeBaseHost::SetUp(){
	job_base_operations_t jbops = JOB_BASE_OPS;
	srand(time(NULL));
	machines = (Machine*)malloc(sizeof(Machine)*MACHINE_AMOUNT);
	for(int i = 0; i < MACHINE_AMOUNT; ++i){
		initMachine(&machines[i]);
		machines[i].base.machine_no = i;
	}

	jobs = (job_t*)malloc(sizeof(job_t) * JOB_AMOUNT);
	for(int i = 0; i < JOB_AMOUNT; ++i){
		initJob(&jobs[i]);
		jobs[i].base.job_no = i;
		jbops.set_process_time(&jobs[i].base, NULL, rand() % 100 + (MACHINE_AMOUNT >> 1));
	}
	
	chromosomes = (Chromosome*)malloc(sizeof(Chromosome) * (CHROMOSOME_AMOUNT << 1));
	double *genes = (double *)malloc(sizeof(double) * (JOB_AMOUNT  << 1) * (CHROMOSOME_AMOUNT << 1)); // genes' memory
	// printf("genes memory usage: %d\n",  (JOB_AMOUNT<<1) * (CHROMOSOME_AMOUNT<<1));
	if(!genes){
		perror("Fatal in allocating genes\n");
		exit(-1);
	}
	for(int i = 0; i < CHROMOSOME_AMOUNT; ++i){
		// printf("i = %d segment = %d ~ %lu\n", i, i*(JOB_AMOUNT<<1), (i+1)*sizeof(double)*(JOB_AMOUNT<<1));
		chromosomes[i].val = i;
		chromosomes[i].base.gene_size = JOB_AMOUNT << 1;
		chromosomes[i].base.chromosome_no = i;
		chromosome_base_init(&chromosomes[i].base, genes + i *(JOB_AMOUNT<<1));
		random_shuffle(&chromosomes[i].base);
	}
}

void TestChromosomeBaseHost::TearDown(){
	free(machines);
	free(jobs);
	free(chromosomes);
}

TEST_F(TestChromosomeBaseHost, test_machine_selection_and_sorting){
	int machine_no;
	list_operations_t ops = LINKED_LIST_OPS;
	machine_base_operations_t mbops = MACHINE_BASE_OPS;
	job_base_operations_t jbops = JOB_BASE_OPS;
	for(int i = 0; i < GENERATIONS; ++i){
		// evaluate the fitness value of each chromosome
#ifdef DEBUG
		printf("Generation %d\n", i);
#endif
		for(int j = 0; j < CHROMOSOME_AMOUNT; ++j){
			// machine selection
#ifdef DEBUG
			printf("%d\tMachine Selection...\n", j);
#endif
			for(int k = 0; k < JOB_AMOUNT; ++k){
				// printf("j = %d, k = %d\n",j, k);
				// link chromosome and jobs
				jbops.set_ms_gene_addr(&jobs[k].base, chromosomes[j].base.ms_genes + k);
				jbops.set_os_gene_addr(&jobs[k].base, chromosomes[j].base.os_genes + k);
				
				// machine selection part1
				machine_no = jobs[k].base.machine_no = jbops.machine_selection(&jobs[k].base);
				
				// machine selection part2
				// machines[machine_no].base.addJob(&machines[machine_no].base, &jobs[k]);
				mbops.add_job(&machines[machine_no].base, &jobs[k].ele);
			}

			// sorting
#ifdef DEBUG
			printf("%d\tSorting...\n",j);	
#endif
			for(int k = 0; k < MACHINE_AMOUNT; ++k){
				mbops.sort_job(&machines[k].base, &ops);
				mbops.reset(&machines[k].base);
				// machines[k].base.sortJob(&machines[k], &ops);
				// machines[k].base.reset(&machines[k].base);
			}
			random_shuffle(&chromosomes[j].base);
		}
	}
}
