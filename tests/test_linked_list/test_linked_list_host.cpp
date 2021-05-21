#include <include/linked_list.h>
#include <gtest/gtest.h>
#include <iostream>
#include <tests/include/test_linked_list.h>
#include <tests/include/def.h>

#undef amount
#define amount 5000

using namespace std;



class TestLinkedListHost : public testing::Test{
public:
	list_ele_t ** eles;
	list_item_t ** eles_arr;
	int values[amount][amount];
	int sizes[amount];
	
	void SetUp() override;
	void TearDown() override;
};

void TestLinkedListHost::SetUp(){
	eles = (list_ele_t**)malloc(sizeof(list_ele_t *) * amount);
	for(int i = 0; i < amount; ++i){
		eles[i] = list_ele_new();
	}

	eles_arr = (list_item_t **)calloc(amount, sizeof(list_item_t *));
	int nums, val;
	int count = 0;	
	for(int i = 0; i < amount; ++i){
		nums = rand() % 10;
		sizes[i] = nums;
		for(int j = 0; j < nums; ++j){
			val = rand() % 1024;
            list_item_add(&eles_arr[i], new_list_item(val), LINKED_LIST_OPS);
			values[i][j] = val;
			count ++;
		}
	}
	PRINTF("Amount of Linked list is %d\n", count);
}

void TestLinkedListHost::TearDown(){
	if(eles){
		for(int i = 0; i < amount; ++i)
			free(eles[i]);
		// free(eles);
		eles = NULL;
	}
		
	// list_item_t *p;
	// list_ele_t *n;
	// if(eles_arr){
	// 	for(int i = 0; i < amount; ++i){
	// 		for(n = &eles_arr[i]->ele; n;){
	// 			p = (list_item_t *)n->pDerivedObject;
	// 			n = n->next;
	// 			free(p);
	// 		}
	// 	}
	// 	// free(eles_arr);
	// 	eles_arr = NULL;

	// }
}


// Test 
TEST_F(TestLinkedListHost, test_set_next_on_host){
	list_operations_t ops = LINKED_LIST_OPS;
	for(int i = 0, range = amount - 1; i < range; ++i){
		ops.set_next(eles[i], eles[i + 1]);
		// eles[i]->set_next(eles[i], eles[i + 1]);
	}

	for(int i = 0, range = amount - 1; i < range; ++i){
		ASSERT_EQ(eles[i]->next, eles[i + 1]);
	}

	for(int i = 1; i < amount; ++i){
		ASSERT_EQ(eles[i]->prev, eles[i - 1]);
	}
}

TEST_F(TestLinkedListHost, test_set_prev_on_host){
	list_operations_t ops = LINKED_LIST_OPS;
	for(int i = 1; i < amount; ++i){
		ops.set_prev(eles[i], eles[i - 1]);
		// eles[i]->set_prev(eles[i], eles[i - 1]);
	}

	for(int i = 1; i < amount; ++i){
		ASSERT_EQ(eles[i]->prev, eles[i - 1]);
	}

	for(int i = 0, range = amount - 1; i < range; ++i){
		ASSERT_EQ(eles[i]->next, eles[i + 1]);
	}
}

TEST_F(TestLinkedListHost, test_sort_linked_list_on_host){
	list_ele_t * iter;
	list_operations_t ops = LINKED_LIST_OPS;
	// list_ele_t test;
	for(int i = 0; i < amount; ++i){
		qsort(values[i], sizes[i], sizeof(int), cmpint);
		if(sizes[i] != 0){
			iter = list_merge_sort(&(eles_arr[i]->ele), &ops);
			eles_arr[i] = (list_item_t *)iter->ptr_derived_object;
			iter = &eles_arr[i]->ele;
			// printf("Value : ");
			for(int j = 0; j < sizes[i]; ++j){
				// printf("%.2f ", iter->get_value(iter));
				ASSERT_EQ(iter->get_value(iter), values[i][j]);
				iter = iter->next;
			}
		}
	}
}
