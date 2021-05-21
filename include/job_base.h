/**
 * @file job_base.h
 * @brief job object definition and functions
 *
 * The file defines job_base_t type and its related functions and also the
 * sturcture of process time.
 *
 * job_base_t is an algorithm object in genetic algorithm. The genes in
 * job_base_t determine what kind of machine is used to work on the job and the
 * order in the machine. job_base_t is also used to record a job and its basic
 * information such as quantity, arrival time, start working time, the end of
 * working time and etc. job_base_t can also be embedded in a container
 * structure.
 *
 * job_base_t's related function type is defined in job_base_operations_t. The
 * variable of job_base_operations_t are function pointer which pointed on the
 * functions to perform the function on job_base_t object. If user does not wat
 * to define the operations, user could use default job_base_operations_t
 * initializer, JOB_BASE_OPS, to initialize the job_base_operations_t object and
 * also use the provided functions.
 *
 * In this file, process_time_t is also defined. process_time_t is an structure
 * to record the machine no and its process time.
 */

#ifndef __JOB_BASE_H__
#define __JOB_BASE_H__

#include <include/def.h>
#include <include/linked_list.h>
#include <stddef.h>


typedef struct process_time_t process_time_t;
typedef struct job_base_t job_base_t;
typedef struct job_base_operations_t job_base_operations_t;

/**
 * @struct process_time_t
 * @brief A structure that store machine number and its process time
 *
 * process_time_t is a structure which store machine number and its process
 * time. All data is in numeric type such as unsigned int and double.
 * process_time_t object can be embedded in a container structure. @b
 * ptr_derived_object maintains the relationship between process_time_t object
 * and the container structure object.
 *
 * @var machine_no : variable store the machine number
 * @var process_time : double type which store the process time of @b machine_no
 * @var ptr_derived_object : @b void* type pointer which point the the container
 * structure object.
 */
struct process_time_t {
    unsigned int machine_no;
    double process_time;
    void *ptr_derived_object;
};


/**
 * @struct job_base_t
 * @brief A structure that store genes and some basic information about job
 *
 * job_base_t is a genetic algorithm object which store genes and some information about a job.
 * @b ms_gene determines which machine accept this job. @b os_seq_gene determines the order of accepted job in machine. Both variables are double const * type to prevent setting value on gene. 
 */
struct job_base_t {
    void *ptr_derived_object;

    // genes point to chromosome's gene
    // use double const * type to prevent set the wrong value on gene
    double const *ms_gene;
    double const *os_seq_gene;

    // partition is the partition value of roulette.
    // for example : size of can run tools is 10, partition is 1/10
    double partition;

    // process time
    // process_time is an 1-D array
    process_time_t *process_time;
    unsigned int size_of_process_time;

    // job information
    unsigned int job_no;
    unsigned int qty;
    unsigned int machine_no;
    double arriv_t;
    double start_time;
    double end_time;
};

struct job_base_operations_t {
    // constructor and initialization
    void (*init)(void *self);
    void (*reset)(job_base_t *self);

    // setter
    void (*set_ms_gene_addr)(job_base_t *self, double *ms_gene);
    void (*set_os_gene_addr)(job_base_t *self, double *os_seq_gene);
    void (*set_process_time)(job_base_t *self,
                             process_time_t *,
                             unsigned int size_of_process_time);
    void (*set_arrival_time)(job_base_t *self, double arrivT);
    void (*set_start_time)(job_base_t *self, double startTime);

    // getter
    double (*get_ms_gene)(job_base_t *self);
    double (*get_os_gene)(job_base_t *self);
    double (*get_arrival_time)(job_base_t *self);
    double (*get_start_time)(job_base_t *self);
    double (*get_end_time)(job_base_t *self);
    unsigned int (*get_machine_no)(job_base_t *self);

    // operation
    unsigned int (*machine_selection)(job_base_t *self);
};

job_base_t *job_base_new();
__qualifier__ void job_base_init(void *self);
__qualifier__ void job_base_reset(job_base_t *self);
__qualifier__ void set_ms_gene_addr(job_base_t *self, double *ms_gene);
__qualifier__ void set_os_gene_addr(job_base_t *self, double *os_seq_gene);
__qualifier__ void set_process_time(job_base_t *self,
                                    process_time_t *pt,
                                    unsigned int size_of_process_time);
__qualifier__ void set_arrival_time(job_base_t *self, double arrivT);
__qualifier__ void set_start_time(job_base_t *self, double startTime);
__qualifier__ double get_ms_gene(job_base_t *self);
__qualifier__ double get_os_gene(job_base_t *self);
__qualifier__ double get_arrival_time(job_base_t *self);
__qualifier__ double get_start_time(job_base_t *self);
__qualifier__ double get_end_time(job_base_t *self);
__qualifier__ unsigned int get_machine_no(job_base_t *self);
__qualifier__ unsigned int machine_selection(job_base_t *self);

#ifndef JOB_BASE_OPS
#define JOB_BASE_OPS                                                      \
    job_base_operations_t                                                 \
    {                                                                     \
        .init = job_base_init, .reset = job_base_reset,                   \
        .set_ms_gene_addr = set_ms_gene_addr,                             \
        .set_os_gene_addr = set_os_gene_addr,                             \
        .set_process_time = set_process_time,                             \
        .set_arrival_time = set_arrival_time,                             \
        .set_start_time = set_start_time, .get_ms_gene = get_ms_gene,     \
        .get_os_gene = get_os_gene, .get_arrival_time = get_arrival_time, \
        .get_start_time = get_start_time, .get_end_time = get_end_time,   \
        .get_machine_no = get_machine_no,                                 \
        .machine_selection = machine_selection                            \
    }
#endif

#endif
