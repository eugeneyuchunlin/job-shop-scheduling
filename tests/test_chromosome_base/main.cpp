#include <gtest/gtest.h>

int JOB_AMOUNT = 20000;
int MACHINE_AMOUNT = 1000;
int CHROMOSOME_AMOUNT = 800;
int GENERATIONS = 1;

int main(int argc, char ** argv){
	testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	return ret;
}
