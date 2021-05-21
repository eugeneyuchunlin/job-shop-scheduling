#include <include/linked_list.h>
	
typedef struct list_item list_item_t;

struct list_item{
	list_ele_t ele;
	double val;
};


// create a list item
list_item_t * new_list_item(double val);

// get the value of list_item
double list_item_get_value(void * ele);



