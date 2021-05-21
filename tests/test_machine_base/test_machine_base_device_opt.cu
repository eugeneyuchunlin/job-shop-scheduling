#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <include/linked_list.h>
#include <include/machine_base.h>
#include <include/job_base.h>
#include <tests/include/def.h>

#include <tests/include/test_machine_base.h>

#define amount 5000

class TestMachineBaseDevice : public testing::Test{
public:
	int **device_values;
	int **address_of_values;

	int **result_device;
	int **result_device_arr;
	
	// Machines
	Machine **machines_device_addresses;
	Machine **device_machines;

	// Jobs
	job_t **jobs_device_addresses;
	job_t **job_device;
	
	unsigned int *sizes;

	void SetUp()override;
	void TearDown()override;
};

void TestMachineBaseDevice::SetUp(){
	// 
}

void TestMachineBaseDevice::TearDown() {

}
