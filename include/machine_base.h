/**
 * @file machine_base.h
 * @brief machine object definition and funcitions
 *
 * The file defines machine_base_t type and its related
 * functions. machine_base_t is used to record the set of jobs
 * and the sequence of jobs. machine_base_t also record some basic
 * information about the machine. The detail information is in
 * machine_base_t.
 *
 * The machine_base_t's related function type is defined in
 * machine_base_operations_t. The variables of machine_base_operations_t
 * are function pointer which pointed on the functions to perform the
 * function on machine_base_t object. If user does not want to define
 * the operations, user could use default machine_base_operations_t
 * initilizer, MACHINE_BASE_OPS, to initialize the machine_base_operations_t
 * object and also use the provided functions.
 *
 * Happy coding!
 *
 * @author Eugene Lin <lin.eugene.l.e@gmail.com>
 * @date 2021.4.30
 */
#ifndef __MACHINE_BASE_H__
#define __MACHINE_BASE_H__

#include <include/def.h>
#include <include/job_base.h>
#include <include/linked_list.h>
#include <stddef.h>

#if defined __NVCC__ || defined __cplusplus
extern "C" {
#endif

typedef struct machine_base_t machine_base_t;

machine_base_t *machine_base_new(unsigned int machine_no);

/**
 * @struct machine_base_t
 * @brief A machine_base object in genetic algorithm
 *
 * machine_base_t object is used to record its jobs and the job sequence.
 * The job sequence is formed by linked list. Linked list is convenient to
 * insert job in the sequence. @b root is the head of the job sequence.
 * @b tail point on the tail of job sequence.
 *
 * machine_base_t object isn't only record its jobs but basic information
 * about machine such as @b machine_no and @b availiable_time (recover time)
 *
 * machine_base_t object can be embedded in a container structure.
 * @b ptr_derived_object maintains the relationship, which are pointed to
 * other container structure if it contains @b machine_base_t object
 * so that machine_base_t can get more information of container structure.
 *
 * @var ptr_derived_object : pointer to parent object
 * @var root : point on the head of jobs
 * @var tail : point on the tail of jobs
 * @var machine_no : the machine's number
 * @var size_of_jobs : the amount of jobs
 * @var avaliable_time : the time the machine can start being used.
 */
struct machine_base_t {
    /// pointer to parent object
    void *ptr_derived_object;

    /// point on the head of jobs
    list_ele_t *root;

    /// point on the tail of jobs
    list_ele_t *tail;

    /// the machine's number
    unsigned int machine_no;

    /// the amount of jobs
    unsigned int size_of_jobs;

    /// the time the machine can start being used.
    unsigned int avaliable_time;
};

/**
 * @struct machine_base_operations_t
 * @brief The structure to store all operations of struct machine_base_t.
 *
 * structure machine_base_operations_t is used to link to the functions, which
 * we would like to perform on machine_base_t object, on host or on device. User
 * can define their own operations which user would like to perform on
 * machine_base_t object. The variables of machine_base_operations_t are
 * function pointers.
 *
 * User can use default machine_base_operations_t initializer, MACHINE_BASE_OPS,
 * to set machine_base_operations_t object to point on the functions provided by
 * this  library.
 *
 * @b set_up_times has an incomplete type declaration which can be use to store
 * lots of functions to evaluate the setup time between two jobs.
 *
 * @var init : pointer to a function to initialize object
 * @var reset : pointer to a function to reset object
 * @var add_job : pointer to a function which perform adding job to machine
 * @var sort_job : pointer to a function which is used to sort the jobs.
 *
 * @var get_quality : pointer to a function which is used to evaluate the
 * quality of machine
 *
 * @var get_size_of_jobs : pointer to a function which return the amount of
 * jobs of machine
 *
 * @var get_setup_time : pointer to a function which is used to evaluate the
 * setup time betweentwo jobs. The setup time functions are in set_up_times in
 * passed parameter ops.
 *
 * @var sizeof_setup_time_function_array : record the amount of setup time
 * functions
 * @var set_up_times : pointers to functions which are used to compute the setup
 * time between two jobs.
 *
 */
struct machine_base_operations_t {
    /// pointer to a function to initialize object
    void (*init)(void *self);

    /// pointer to a function to reset object
    void (*reset)(machine_base_t *self);

    /// pointer to a function which perform adding job to machine
    void (*add_job)(machine_base_t *self, list_ele_t *);

    /// pointer to a function which is used to sort the jobs.
    void (*sort_job)(machine_base_t *self, list_operations_t *ops);

    /// pointer to a function which is used to evaluate the quality of the
    /// machine
    void (*get_quality)(machine_base_t *self);

    /// pointer to a function which return the amount of jobs of machine
    unsigned int (*get_size_of_jobs)(machine_base_t *self);

    /// pointer to a function which is used to evaluate the
    /// setup time betweentwo jobs. The setup time functions are in set_up_times
    /// in passed parameter ops.
    double (*get_setup_times)(machine_base_operations_t *ops,
                              job_base_t *job1,
                              job_base_t *job2);


    /// sizeof_setup_time_function_array : record the amount of setup time
    /// functions
    size_t sizeof_setup_time_function_array;

    /// pointers to functions which are used to compute the setup time between
    /// two jobs.
    double (*set_up_times[])(job_base_t *job1, job_base_t *job2);
};

/**
 * machine_base_reset () - Reset machine_base_t object
 * Reset machine base object. Clean the jobs.
 * @param _self : machine base object
 */
__qualifier__ void machine_base_reset(machine_base_t *_self);

/**
 * machine_base_get_size_of_jobs () - Return the amount of jobs of machine base
 * object
 * @param _self : the machine base object
 */
__qualifier__ unsigned int machine_base_get_size_of_jobs(machine_base_t *_self);

/**
 * machine_base_init () - Initialize machine base object
 * @param _self : machine base object
 */
__qualifier__ void machine_base_init(machine_base_t *_self);

/**
 * _machine_base_add_job () - Add new job into machine
 * Add new job into machine. The data structure used to store jobs is linked
 * list.
 * @b job should be list_ele_t type. The way to form the job linked list could
 * be re-defined. _machine_base_add_job provide default method to form the
 * linked list. The new job would be at the end of list.
 * @param _self : machine_base object
 * @param job : new job
 */
__qualifier__ void _machine_base_add_job(machine_base_t *_self,
                                         list_ele_t *job);

/**
 * _machine_base_sort_job - Sort the job sequence
 * Sort the job sequence in order. Merge sort algorithm is adopted to sort the
 * job sequence because its worst case complexity is O(nlogn).
 * @param _self : machine_base object
 * @param ops : operations for list element
 */
__qualifier__ void _machine_base_sort_job(machine_base_t *_self,
                                          list_operations_t *ops);

/**
 * @def MACHINE_BASE_OPS
 * Initialize the machine_base_operations_t object by default function
 * provided by this library.
 */
#ifndef MACHINE_BASE_OPS
#define MACHINE_BASE_OPS                                               \
    machine_base_operations_t                                          \
    {                                                                  \
        .reset = machine_base_reset, .add_job = _machine_base_add_job, \
        .sort_job = _machine_base_sort_job,                            \
        .get_size_of_jobs = machine_base_get_size_of_jobs,             \
    }
#endif

#if defined __NVCC__ || defined __cplusplus
}
#endif

#endif
