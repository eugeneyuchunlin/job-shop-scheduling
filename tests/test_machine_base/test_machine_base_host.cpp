#include <include/def.h>
#include <include/machine_base.h>
#include <gtest/gtest.h>

#include <include/linked_list.h>
#include <tests/include/test_machine_base.h>

#define amount 5000


class TestMachineBaseHost : public testing::Test{
public:
	int ** values;
	int sizes[amount];
	job_t ***jobs;
	Machine **machines;
	void SetUp()override;
	void TearDown() override;
};

void TestMachineBaseHost::TearDown(){
	for(int i = 0; i < amount; ++i){
		free(values[i]);
		free(machines[i]);
		for(int j = 0; j < sizes[i]; ++j)
			free(jobs[i][j]);
		free(jobs[i]);
	}
	free(values);
	free(machines);
	free(jobs);
}

void TestMachineBaseHost::SetUp(){
	jobs = (job_t***)malloc(sizeof(job_t**) * amount);
	values = (int**)malloc(sizeof(int*)*amount);
	machines = (Machine**)malloc(sizeof(Machine*)*amount);

	for(int i = 0; i < amount; ++i){
		sizes[i] = rand() % 1000 + 1;
		values[i] = (int*)malloc(sizeof(int)*sizes[i]);
		jobs[i] = (job_t**)malloc(sizeof(job_t*) * sizes[i]);
		machines[i] = newMachine();
		for(int j = 0; j < sizes[i]; ++j){
			values[i][j] = rand() % 1024;
			jobs[i][j] = newJob(values[i][j]);
		}	
	}

}

TEST_F(TestMachineBaseHost, test_machine_base_host_add_job){
	machine_base_operations_t mbops = MACHINE_BASE_OPS;
	for(int i = 0; i < amount; ++i){
		for(int j = 0; j < sizes[i]; ++j){
			mbops.add_job(&machines[i]->base, &jobs[i][j]->ele);
		}
	}
	
	list_ele_t *ele;
	for(int i = 0; i < amount; ++i){
		ele = machines[i]->base.root;
		for(int j = 0; j < sizes[i]; ++j){
			ASSERT_EQ(ele->get_value(ele), values[i][j]) << "i = " << i << "j = " << j << std::endl;
			ele = ele->next;
		}	
	}
}

TEST_F(TestMachineBaseHost, test_machine_base_host_sort_job){

	list_operations_t ops = LINKED_LIST_OPS;
	machine_base_operations_t mbops = MACHINE_BASE_OPS;
	for(int i = 0; i < amount; ++i){
		for(int j = 0; j < sizes[i]; ++j){
			mbops.add_job(&machines[i]->base, &jobs[i][j]->ele);
		}
	}

	
	for(int i = 0; i < amount; ++i){
		qsort(values[i], sizes[i], sizeof(int), cmpint);
		mbops.sort_job(&machines[i]->base, &ops);
	}

	list_ele_t *ele;
	for(int i = 0; i < amount; ++i){
		ele = machines[i]->base.root;
		for(int j = 0; j < sizes[i]; ++j){
			ASSERT_EQ(ele->get_value(ele), values[i][j]) << "i = " << i << "j = " << j << std::endl;
			ele = ele->next;
		}	
	}
}
