#include <features.h>
#include <include/linked_list.h>
#include <stdio.h>
#include <stdlib.h>
// #define DEBUG
#define show(list)                                                  \
    for (LinkedListElement *iter = list; iter; iter = iter->next) { \
        printf("%.2f ", iter->get_value(iter));                     \
    }                                                               \
    printf("\n");

__qualifier__ void _list_ele_set_next(void *_self, list_ele_t *_next)
{
    list_ele_t *self = (list_ele_t *) _self;
    self->next = _next;
    if (_next)
        _next->prev = self;
}


__qualifier__ void _list_ele_set_prev(void *_self, list_ele_t *_prev)
{
    list_ele_t *self = (list_ele_t *) _self;
    self->prev = _prev;
    if (_prev)
        _prev->next = self;
}

__qualifier__ void _list_init(void *_self)
{
    list_ele_t *self = (list_ele_t *) _self;
    self->next = self->prev = NULL;
    self->get_value = NULL;
}

list_ele_t *list_ele_new()
{
    list_ele_t *ele = (list_ele_t *) malloc(sizeof(list_ele_t));
    if (!ele)
        return ele;
    ele->ptr_derived_object = NULL;
    ele->next = ele->prev = NULL;
    return ele;
}

__qualifier__ list_ele_t *list_merge(list_ele_t *l1,
                                     list_ele_t *l2,
                                     list_operations_t *ops)
{
    if (!l2)
        return l1;
    if (!l1)
        return l2;

    list_ele_t *result, *result_iter;

    // set the first element of result
    double val1 = l1->get_value(l1);
    double val2 = l2->get_value(l2);
    if (val1 < val2) {
        result = l1;
        l1 = l1->next;
    } else {
        result = l2;
        l2 = l2->next;
    }

    result_iter = result;

    // merge the linked list
    while (l1 && l2) {
        if (l1->get_value(l1) < l2->get_value(l2)) {
            ops->set_next(result_iter, l1);
            l1 = l1->next;  // l1 move to next element
        } else {
            ops->set_next(result_iter, l2);
            l2 = l2->next;  // l2 move to next element
        }
        result_iter = result_iter->next;  // point to next element
    }

    // if l1 is not empty, connect to result
    // if(l1) result_iter->set_next(result_iter, l1);
    // else if(l2) result_iter->set_next(result_iter, l2);
    // if(l1) ops.set_next(result_iter, l1);
    // else if(l2) ops.set_next(result_iter, l2);
    if (l1)
        _list_ele_set_next(result_iter, l1);
    else if (l2)
        _list_ele_set_next(result_iter, l2);
    return result;
}


__qualifier__ list_ele_t *list_merge_sort(list_ele_t *head,
                                          list_operations_t *ops)
{
    if (!head || !head->next) {
        return head;
    } else {
        list_ele_t *fast = (list_ele_t *) head->next;
        list_ele_t *slow = head;

        // get the middle of linked list
        // divide the linked list
        while (fast && fast->next) {
            slow = (list_ele_t *) slow->next;
            fast = (list_ele_t *) ((list_ele_t *) fast->next)->next;
        }
        // now, get two lists.
        fast = (list_ele_t *) slow->next;
        fast->prev = NULL;
        slow->next = NULL;
#ifdef DEBUG
        printf("Head : ");
        show(head);
        printf("Fast : ");
        show(fast);
#endif
        list_ele_t *lhs = list_merge_sort(head, ops);
#ifdef DEBUG
        printf("lhs finish!\n");
#endif
        list_ele_t *rhs = list_merge_sort(fast, ops);

        // merge the linked list
#ifdef DEBUG
        list_ele_t *result = mergeLinkedList(lhs, rhs, ops);
        printf("Merge Result : ");
        show(result);
        return result;
#endif

        return list_merge(lhs, rhs, ops);
    }
}
