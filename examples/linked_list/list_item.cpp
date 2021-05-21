#include <stdlib.h>
#include <include/linked_list.h>
#include "list_item.h"

list_item_t * new_list_item(double _val){
	list_item_t *item = (list_item_t*)malloc(sizeof(list_item_t));
	/*TODO: create an list item and the field val equal to _val
	 * if failed on memory allocation, return NULL
	 * hint : item->ele.ptr_derived_object should point to item
	 * */
	return NULL;
}


double list_item_get_value(void *_self){
	list_ele_t * self = (list_ele_t *)_self;
	/*TODO: return the value of list item*/
	/*hint : use ptr_derived_object*/
}

