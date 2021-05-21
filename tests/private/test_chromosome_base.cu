#include <include/chromosome_base.h>
#include <tests/include/test_chromosome_base.h>

Chromosome * createChromosome(size_t gene_size){
	Chromosome * chromosome = (Chromosome *)malloc(sizeof(Chromosome));
	if(!chromosome)
		return NULL;

	chromosome_base_init(&chromosome->base, NULL);
	return chromosome;
}
