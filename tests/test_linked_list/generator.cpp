#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


int main(int argc, char *argv[]){
	srand(time(NULL));
	FILE *file = fopen("data.txt", "w");
	int i, j;
	int amount = 0;
	int row = atoi(argv[1]);
	int col = atoi(argv[2]);
	for(i = 0; i < row; ++i){
		amount = rand() % col;
		for(j = 0; j < amount; ++j){
			fprintf(file, "%.2f,",((float)rand() / (float)RAND_MAX) * 1024);
		}
		fprintf(file, "%.2f\n", ((float)rand() / (float)RAND_MAX) * 1024);
	}
	fclose(file);
}
