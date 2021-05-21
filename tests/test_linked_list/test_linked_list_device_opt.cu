#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gtest/gtest.h>
#include <assert.h>
#include <tests/include/def.h>

#include <include/linked_list.h>
#include <tests/include/test_linked_list.h>


#ifdef STATISTIC
extern int amount;
extern FILE * log_file;
#else 
#define amount 500000
#endif

class TestLinkedListDeviceOpt : public testing::Test{
public:
	int **values;
	int **values_device;
	int **values_arr_device;
	
	list_item_t ** item_address_on_device;

	list_item_t ** items;

	int *sizes;
	int *sizes_device;

	size_t usage;
	int count;

	void SetUp() override;
	void TearDown() override;
};

void TestLinkedListDeviceOpt::SetUp(){
	usage = 0;
	count = 0;
	cudaError_t err;
	err = cudaMallocHost((void**)&values, sizeof(int*)*amount); // alloc values
	assert(err == cudaSuccess);

	err = cudaMallocHost((void**)&sizes, sizeof(int*)*amount); // alloc sizes
	assert(err == cudaSuccess);

	err = cudaMalloc((void**)&sizes_device, sizeof(int)*amount);// alloc sizes_device
	assert(err == cudaSuccess);
	
	err = cudaMallocHost((void**)&item_address_on_device, sizeof(list_item_t*)*amount); // init address_on_device;
	assert(err == cudaSuccess);

	err = cudaMalloc((void**)&items, sizeof(list_item_t*)*amount); // alloc items
	assert(err == cudaSuccess);

	err = cudaMalloc((void**)&values_device, sizeof(int*)*amount); // alloc values_device
	assert(err == cudaSuccess);

	err = cudaMallocHost((void**)&values_arr_device, sizeof(int*)*amount); // alloc values_arr_device on host
	assert(err == cudaSuccess);

	list_item_t *tmp;
	for(int i = 0; i < amount; ++i){
		sizes[i] = rand() % 100+50;

		err = cudaMalloc((void**)&tmp, sizeof(list_item_t)*sizes[i]); // alloc instance
		assert(err == cudaSuccess);

		item_address_on_device[i] = tmp;
	
		err = cudaMallocHost((void**)&values[i], sizeof(int)*sizes[i]); // alloc values
		assert(err == cudaSuccess);
		for(int j = 0; j < sizes[i]; ++j){
			values[i][j] = rand() % 1024;
		}
		count += sizes[i];
	}

	err = cudaMemcpy(items, item_address_on_device, sizeof(list_item_t*)*amount, cudaMemcpyHostToDevice); // cpy items ptr value
	assert(err == cudaSuccess);

	int *values_tmp;
	
	for(int i = 0; i < amount; ++i){
		err = cudaMalloc((void**)&values_tmp, sizeof(int)*sizes[i]);
		assert(err == cudaSuccess);

		err = cudaMemcpy(values_tmp, values[i], sizeof(int)*sizes[i], cudaMemcpyHostToDevice);
		assert(err == cudaSuccess);

		values_arr_device[i] = values_tmp;
	}
	err = cudaMemcpy(values_device, values_arr_device, sizeof(int*)*amount, cudaMemcpyHostToDevice); // cpy values ptr value
	assert(err == cudaSuccess);

	err = cudaMemcpy(sizes_device, sizes, sizeof(int)*amount, cudaMemcpyHostToDevice); // cpy sizes value
	assert(err == cudaSuccess);
}

void TestLinkedListDeviceOpt::TearDown(){
	// free item
	for(int i = 0 ; i < amount; ++i){
		cudaFree(item_address_on_device[i]);
		cudaFree(values_arr_device[i]);
		cudaFreeHost(values[i]);
	}
	cudaFreeHost(sizes);
	cudaFree(sizes_device);

	cudaFreeHost(values);
	cudaFree(values_device);
	cudaFreeHost(values_arr_device);

	cudaFreeHost(item_address_on_device);
	cudaFree(items);
}

__global__ void initLinkedListOpsKernel(list_operations_t * op){
    list_operations_t tmp = LINKED_LIST_OPS;
    *op = tmp;
    op->init = _list_init;
}

__global__ void sortingSetupKernel(list_item_t **items, int **values, int *sizes, unsigned int offset, int am){
	int idx = threadIdx.x + blockIdx.x * blockDim.x + offset;
	if(idx < am){
		list_operations_t ops = LINKED_LIST_OPS;
		for(int i = 0; i < sizes[idx]; ++i){
			items[idx][i].value = values[idx][i];
			_list_init(&items[idx][i].ele);
			items[idx][i].ele.get_value = linkedListItemGetValue;
			items[idx][i].ele.ptr_derived_object = &items[idx][i];
		}
		
		for(int i = 0; i < sizes[idx] - 1; ++i){
			ops.set_next(&items[idx][i].ele, &items[idx][i + 1].ele);
		}
	}
}

__global__ void sorting(list_item_t **items, int **values, list_operations_t *ops, unsigned int offset, int am){
	int idx = threadIdx.x + blockIdx.x * blockDim.x + offset;
	if(idx < am){
		list_ele_t *iter;
		iter = list_merge_sort(&items[idx][0].ele, ops);

		for(int i = 0; iter; ++i){
			values[idx][i] = iter->get_value(iter);
			iter = iter->next;
		}
	}
}

TEST_F(TestLinkedListDeviceOpt, test_sort_linked_list_on_device_opt){
	int ** result_arr;
	int **result_arr_device;
	int *result_tmp;
	
	
	cudaMallocHost((void**)&result_arr, sizeof(int*)*amount);
	for(int i = 0; i < amount; ++i){
		ASSERT_EQ(cudaMalloc((void**)&result_tmp, sizeof(int)*sizes[i]), cudaSuccess);
		usage += sizeof(int)*sizes[i];
		result_arr[i] = result_tmp;
	}
	ASSERT_EQ(cudaMalloc((void**)&result_arr_device, sizeof(int*)*amount), cudaSuccess);
	ASSERT_EQ(cudaMemcpy(result_arr_device, result_arr, sizeof(int*)*amount, cudaMemcpyHostToDevice), cudaSuccess);
	usage += sizeof(int*)*amount;


	/********INIT OPS**********************************/
	list_operations_t *ops_device;
	ASSERT_EQ(cudaMalloc((void**)&ops_device, sizeof(list_operations_t)), cudaSuccess);
	initLinkedListOpsKernel<<<1, 1>>>(ops_device);

	clock_t t1 = clock();
	sortingSetupKernel<<<1024, 1024>>>(items, values_device, sizes_device, 0, amount);
	// sortingSetupKernel<<<1024, 1024>>>(items, values_device, sizes_device, 100000, amount);
	sorting<<<1024, 1024>>>(items, result_arr_device, ops_device, 0, amount);
	// sorting<<<1024, 1024>>>(items, result_arr_device, ops_device, 100000, amount);
	cudaDeviceSynchronize();
	clock_t t2 = clock();
	PRINTF("Testing amount : %d\n", count);
	PRINTF("Device memory usage : %lu bytes\n", usage);
	PRINTF("Time elapse : %.3fs\n", (t2 - t1) / (double)CLOCKS_PER_SEC);
#ifdef STATISTIC
	fprintf(log_file, "%.3f\n", (t2 - t1) / (double)CLOCKS_PER_SEC);
#endif

	ASSERT_EQ(cudaMemcpy(result_arr, result_arr_device, sizeof(int*)*amount, cudaMemcpyDeviceToHost), cudaSuccess);
	for(int i = 0; i < amount; ++i){
		cudaMallocHost((void**)&result_tmp, sizeof(int)*sizes[i]);
		ASSERT_EQ(cudaMemcpy(result_tmp, result_arr[i], sizeof(int)*sizes[i], cudaMemcpyDeviceToHost), cudaSuccess);
		qsort(values[i], sizes[i], sizeof(int), cmpint);
		for(int j = 0; j < sizes[i]; ++j){
			ASSERT_EQ(values[i][j], result_tmp[j]);
		}
		cudaFreeHost(result_tmp);
		cudaFree(result_arr[i]);
	}
	
	
	cudaFreeHost(result_arr);
	cudaFree(result_arr_device);

}
