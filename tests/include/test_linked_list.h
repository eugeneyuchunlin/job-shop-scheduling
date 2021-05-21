#ifndef __TEST_LINKED_LIST_H__
#define __TEST_LINKED_LIST_H__

#include <include/linked_list.h>
#include <include/common.h>
#include <include/def.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct list_item_t list_item_t;
struct list_item_t {
	list_ele_t ele;
	double value;
};

__qualifier__ double linkedListItemGetValue(void *_self);

list_item_t * new_list_item(double val);


void list_item_add(list_item_t **list, list_item_t *item, list_operations_t ops);


#endif
