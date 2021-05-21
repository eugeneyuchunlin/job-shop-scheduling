#include <include/chromosome_base.h>

struct Chromosome{
	int val;	
	chromosome_base_t base;
};

Chromosome * createChromosome(size_t gene);
