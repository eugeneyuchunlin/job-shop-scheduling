#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <include/linked_list.h>
#include <include/common.h>
#include "list_item.h"
#define amount 15

#define showList(iter, head)                                                 \
	iter = head;                                                             \
	while(iter) {printf("%.0f ", iter->get_value(iter)); iter = iter->next;}  \
	printf("\n");                                                            \

int main(int argc, const char *argv[]){
	
	double value[amount];
	for(int i = 0; i < amount; ++i){
		value[i] = rand() % 100;
	}

	list_operations_t ops = LINKED_LIST_OPS;
	
	list_ele_t  *head = NULL;
	list_item_t *prev;
	for(int i = 0; i < amount; ++i){
		prev = new_list_item(value[i]);
		ops.set_next(&prev->ele, head);
		head = &prev->ele;
	}

	list_ele_t *iter = head;
	printf("Unsorted : ");
	showList(iter, head);
	

	head = list_merge_sort(head, &ops);

	printf("Sorted : ");
	showList(iter, head);

	qsort(value, amount, sizeof(double), cmpdouble);
	iter = head;
	for(int i = 0; i < amount; ++i){
		assert(value[i] == iter->get_value(iter));
		iter = iter->next;
	}
}
