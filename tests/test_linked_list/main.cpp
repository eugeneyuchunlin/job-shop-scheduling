#include <gtest/gtest.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef amount
#undef amount
#endif

#ifdef STATISTIC
int amount;
FILE *log_file;
#endif

int main(int argc, char ** argv){
	testing::InitGoogleTest(&argc, argv);
#ifdef STATISTIC
	int ret;
	log_file = fopen("log.tmp", "w");
	if(!log_file) {
		return -1;
	}

	for(int i = 10000; i < 100000; i += 1000){
		amount = i;
		fprintf(log_file, "%d,", i);
		ret = RUN_ALL_TESTS();
	}
	fclose(log_file);
	return ret;
#endif
	return RUN_ALL_TESTS();
}
