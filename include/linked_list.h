/**
 * @file linked_list.h
 * @brief linked list definition and functions
 *
 * This file defines the data structure of linked list and related functions.
 * The type of node of linked list is struct list_ele_t. struct list_ele_t
 * contains several variables which are used to link list element together or
 * maintain the relationship between element and the object contains the
 * element. The detail infor- mation is in list_ele_t.
 *
 * The funtions which may be performed on list element are collected in
 * struct list_operations_t. The detail information is in list_operations_t.
 *
 *
 * @author Eugene Lin <lin.eugene.l.e@gmail.com>
 * @date 2021.2.23
 */
#ifndef __LINKED_LIST_H__
#define __LINKED_LIST_H__

#include <include/def.h>
#include <stddef.h>


#if defined __NVCC__ || defined __cplusplus
extern "C" {
#endif


typedef struct list_ele_t list_ele_t;

typedef struct list_operations_t list_operations_t;


/**_list_ele_set_next () - Default operation on struct list_ele_t.
 * The function is used to connect next list_ele_t. The two objects will be
 * linked doubly.
 * @param _self : current list element
 * @param _next : next list element
 */
__qualifier__ void _list_ele_set_next(void *_self, list_ele_t *_next);

/**_list_ele_set_prev () - Default operation on struct list_ele_t.
 * The function is used to connect previous list_ele_t. The two objects will be
 * linked doubly.
 * @param _self : current list element
 * @param _prev : previous list element
 */
__qualifier__ void _list_ele_set_prev(void *_self, list_ele_t *_prev);


/**
 * _list_init () - Initialize the linked list element
 * In the function, the pointers of list element will be set NULL
 * @param _self : the element which is going to be initialized
 */
__qualifier__ void _list_init(void *_self);


/**
 * list_merge_sort () - Sort the linked list by using merge sort algorithm
 * @param head: The head of linked list which is going to be sorted. The head is
 * struct list_ele_t * type
 * @param ops: The basic operation on list element. ops should be struct
 * list_ele_t * type.
 * @return head of list
 * @warning
 *  1. ops should not be NULL or the function will crash.
 *  2. The function will use get_value function pointer to evaluate each
 * list_ele_t's value. Please make sure that get_value is not NULL and it point
 * to the correct function before invoking this function.
 *  3. The function will use set_next function pointer to add new list node or
 * concatenate two lists. Please make sure that set_next in @ops is point to
 * correct function.
 *  */
__qualifier__ list_ele_t *list_merge_sort(list_ele_t *head,
                                          list_operations_t *ops);

/**
 * list_merge () - Merge two linked lists and return the result.
 * Merge two linked lists in increasing order.
 * @param l1: a list should be merged
 * @param l2: a list should be merged
 * @param ops: operation performed on list_ele_t object
 */
__qualifier__ list_ele_t *list_merge(list_ele_t *l1,
                                     list_ele_t *l2,
                                     list_operations_t *ops);

/**
 * list_ele_new() - create a list_ele_t object
 * The memory of object is allocated on heap. If fail on memory allocation the
 * function return NULL or the function will will initialize the object and
 * return.
 * @return NULL or list_ele_t object
 */
list_ele_t *list_ele_new();


/**
 * @struct list_ele_t
 * @brief A node of double-linked list
 *
 * Each node contains basically 4 pointers.
 * @b next point to next node of current node. @b prev point to previous node in
 * linked list.
 * @b ptr_drived_object is a @b void* type pointer, which can be used to point
 * to its parent object The operations on list_ele_t object have a structure to
 * maintain the function, the structure is LinkedListElementOpreation.
 * list_operations_t allows user to design their own operations on list_ele_t
 * object.
 *
 * The list nodes are usually embedded in a container structure. @b
 * ptr_derived_object is a @b void* type which can be used to point to the
 * address of container structure that help to manipulate structure.
 * @var next : pointer to next node in linked list
 * @var prev : pointer to previous node in linked list
 * @var ptr_derived_object : pointer to parent object
 */
struct list_ele_t {
    /// pointer to next node in linked list
    list_ele_t *next;

    /// pointer to previous node in linked list
    list_ele_t *prev;

    /// pointer to parent object
    void *ptr_derived_object;

    /// @brief function pointer to a function which is to evaluate the value of
    /// this node. The user must let the function pointer point to the correct
    /// function before invoking the function.
    double (*get_value)(void *self);
};


/**
 * @struct list_operations_t
 * @brief The structure to store all operations of struct list_ele_t. The user
 * can define their own operations.
 *
 * The list_operations_t is used to link to the functions, which we would like
 * to perform on list element, on host or on device. The library provides
 * flexiblility on changing the function which user would like to perform on
 * list element. The variables of list_operations_t are function pointers.
 *
 * @var init : pointer to a function to initialize the list_ele_t
 * @var rest : pointer to a function to reset the list_ele_t
 * @var set_next : pointer to function to set the next node of a list_ele_t
 * @var set_prev : pointer to function to set the previousnode of a list_ele_t
 */
struct list_operations_t {
    /// pointer to a function to initialize the list_ele_t
    void (*init)(void *self);

    /// pointer to a function to reset the list_ele_t
    void (*reset)(void *self);

    /// pointer to function to set the next node of a list_ele_t
    void (*set_next)(void *self, list_ele_t *next);

    /// pointer to function to set the previous node of a list_ele_t
    void (*set_prev)(void *self, list_ele_t *prev);
};


#ifndef LINKED_LIST_OPS
/**
 * @def LINKED_LIST_OPS
 * @brief LINKED_LIST_OPS will create default list_operations_t.
 * in which, the field set_next point to _list_ele_set_next and the field
 * set_prev point to _list_ele_set_prev
 */
#define LINKED_LIST_OPS                                                 \
    list_operations_t                                                   \
    {                                                                   \
        .set_next = _list_ele_set_next, .set_prev = _list_ele_set_prev, \
    }
#endif



#if defined __NVCC__ || defined __cplusplus
}
#endif

#endif
