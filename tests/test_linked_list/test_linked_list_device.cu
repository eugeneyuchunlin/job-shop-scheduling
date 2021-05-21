#include <cstdio>
#include <ctime>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <include/linked_list.h>
#include <gtest/gtest.h>
#include <cuda.h>
#include <regex.h>
#include <texture_types.h>
#include <time.h>
#include <tests/include/def.h>

#define amount 100000

#include <tests/include/test_linked_list.h>

class TestLinkedListDevice : public testing::Test{
public:
	int *values[amount];

	int ** result_arr;
	int **result_arr_device;

	int **values_arr;
	int **values_arr_device;

	list_item_t ***item_address_on_device;
	list_item_t *** items_array_of_array;
	list_item_t ***items;

	size_t usage;


	int sizes[amount];
	int *sizes_device;
	void SetUp() override;
	void TearDown() override;

	void advanceSetup();

};

void TestLinkedListDevice::SetUp(){

}



void TestLinkedListDevice::advanceSetup(){
	item_address_on_device = (list_item_t ***)malloc(sizeof(list_item_t**)*amount);

	list_item_t *item_device;
	int count = 0;
	usage = 0;
	for(int i = 0; i < amount; ++i){
		sizes[i] = rand() % 100 + 50;
		values[i] = (int*)malloc(sizeof(int)*sizes[i]);
		count += sizes[i];
		item_address_on_device[i] = (list_item_t **)malloc(sizeof(list_item_t*)*sizes[i]);
		for(int j = 0; j < sizes[i]; ++j){
			values[i][j] = rand() % 1024;
			ASSERT_EQ(cudaMalloc((void**)&item_device, sizeof(list_item_t)), cudaSuccess);
			item_address_on_device[i][j] = item_device;
		}
	}
	
	usage += count*sizeof(list_item_t);
	
	list_item_t ** items_array;
	items_array_of_array = (list_item_t ***)malloc(sizeof(list_item_t **)*amount);
	for(int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMalloc((void**)&items_array, sizeof(list_item_t*)*sizes[i]), cudaSuccess);
		ASSERT_EQ(cudaMemcpy(items_array, item_address_on_device[i], sizeof(list_item_t*)*sizes[i], cudaMemcpyHostToDevice), cudaSuccess);
		items_array_of_array[i] = items_array;

		usage += sizeof(list_item_t*)*sizes[i];
	}

	ASSERT_EQ(cudaMalloc((void**)&items, sizeof(list_item_t**)*amount), cudaSuccess);
	ASSERT_EQ(cudaMemcpy(items, items_array_of_array, sizeof(list_item_t**)*amount, cudaMemcpyHostToDevice), cudaSuccess);

	usage += sizeof(list_item_t**)*amount;
	PRINTF("Amount of testing elements is %d\n", count);
	PRINTF("Average amount of elements handled by a thread is %.2f\n", count / (double)amount);

	int *result_tmp;
	result_arr = (int**)malloc(sizeof(int*)*amount);
	for(int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMalloc((void**)&result_tmp, sizeof(int)*sizes[i]), cudaSuccess);
		usage += sizeof(int)*sizes[i];
		result_arr[i] = result_tmp;
	}
	ASSERT_EQ(cudaMalloc((void**)&result_arr_device, sizeof(int*)*amount), cudaSuccess);
	ASSERT_EQ(cudaMemcpy(result_arr_device, result_arr, sizeof(int*)*amount, cudaMemcpyHostToDevice), cudaSuccess);
	usage += sizeof(int*)*amount;

	ASSERT_EQ(cudaMalloc((void**)&sizes_device, sizeof(int)*amount), cudaSuccess);
	ASSERT_EQ(cudaMemcpy(sizes_device, sizes, sizeof(int)*amount, cudaMemcpyHostToDevice), cudaSuccess);
	
	usage += sizeof(int)*amount;

	/*********ALLOCATE values array and copy value********/
	values_arr = (int **)malloc(sizeof(int*)*amount);
	int *values_tmp;
	for(int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMalloc((void**)&values_tmp, sizeof(int)*sizes[i]), cudaSuccess);
		ASSERT_EQ(cudaMemcpy(values_tmp, values[i], sizeof(int)*sizes[i], cudaMemcpyHostToDevice), cudaSuccess);
		values_arr[i] = values_tmp;
		usage += sizeof(int)*sizes[i];
	}
	ASSERT_EQ(cudaMalloc((void**)&values_arr_device, sizeof(int*)*amount), cudaSuccess);
	ASSERT_EQ(cudaMemcpy(values_arr_device, values_arr, sizeof(int*)*amount, cudaMemcpyHostToDevice), cudaSuccess);
	usage += sizeof(int*)*amount;

	PRINTF("Device Memory usage : %lu bytes\n", usage);
}

void TestLinkedListDevice::TearDown(){
	// free item
	for(int i = 0; i < amount; ++i){
		for(int j = 0; j < sizes[i]; ++j)
			cudaFree(item_address_on_device[i][j]);
		free(item_address_on_device[i]);
		cudaFree(items_array_of_array[i]);
	}
	free(items_array_of_array);
	free(item_address_on_device);
	cudaFree(items);

	// free result
	for(int i = 0; i < sizes[i]; ++i){
		cudaFree(result_arr[i]);
	}
	cudaFree(result_arr_device);
	free(result_arr);

	// free sizes
	cudaFree(sizes_device);

	// free values
	for(int i = 0; i < amount; ++i){
		cudaFree(values_arr[i]);
	}
	cudaFree(values_arr_device);
	free(values_arr);
	
}

__global__ void initLinkedListOps(list_operations_t *ops){
    list_operations_t tmp = LINKED_LIST_OPS;
    *ops = tmp;
}

__global__ void sortingSetUp(list_item_t ***items,  int *sizes, int **values){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx < amount){
		// first initial all items;
		// connect to device function
		list_operations_t ops = LINKED_LIST_OPS;
		for(int i = 0; i < sizes[idx]; ++i){
			items[idx][i]->ele.get_value = linkedListItemGetValue;
			// items[idx][i]->ele.set_next = _list_ele_set_next;
			// items[idx][i]->ele.set_prev = _list_ele_set_prev;
			items[idx][i]->ele.ptr_derived_object = items[idx][i];
			items[idx][i]->value = values[idx][i];
		}

		for(int i = 0, size = sizes[idx] - 1; i < size; ++i){
			ops.set_next(&items[idx][i]->ele, &items[idx][i + 1]->ele);
			// items[idx][i]->ele.set_next(&items[idx][i]->ele, &items[idx][i + 1]->ele);
		}

	}
}

__global__ void sorting(list_item_t ***items, int **values, list_operations_t *ops, int am){
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
 	if(idx < am){
		list_ele_t *iter;
 		iter = list_merge_sort(&(items[idx][0]->ele), ops);
		items[idx][0] = (list_item_t*)iter->ptr_derived_object;
		iter = &(items[idx][0]->ele);

		for(int i = 0; iter ; ++i){
			values[idx][i] = iter->get_value(iter);
			iter = iter->next;
		}
 	}
}



TEST_F(TestLinkedListDevice, test_sort_linked_list_on_device){

	/**********ALLOCAT result array***********/
	advanceSetup();
	/********INIT OPS**********************************/
	list_operations_t *ops_device;
	ASSERT_EQ(cudaMalloc((void**)&ops_device, sizeof(list_operations_t)), cudaSuccess);
	initLinkedListOps<<<1,1>>>(ops_device);
	

	clock_t t1 = clock();
	sortingSetUp<<<1024, 1024>>>(items, sizes_device, values_arr_device);
	sorting<<<1024, 1024>>>(items, result_arr_device, ops_device, amount);
	cudaDeviceSynchronize();
	clock_t t2 = clock();
	PRINTF("Time elapse : %.3fs\n", (t2 - t1) / (double)CLOCKS_PER_SEC);
	
	int *result_tmp;
	ASSERT_EQ(cudaMemcpy(result_arr, result_arr_device, sizeof(int*)*amount, cudaMemcpyDeviceToHost), cudaSuccess);
	for(int i = 0; i < amount; ++i){
		result_tmp = (int*)malloc(sizeof(int)*sizes[i]);
		ASSERT_EQ(cudaMemcpy(result_tmp, result_arr[i], sizeof(int)*sizes[i], cudaMemcpyDeviceToHost), cudaSuccess);
		qsort(values[i], sizes[i], sizeof(int), cmpint);
		for(int j = 0; j < sizes[i]; ++j){
			ASSERT_EQ(values[i][j], result_tmp[j]); 
		}
		free(result_tmp);
	}

}
