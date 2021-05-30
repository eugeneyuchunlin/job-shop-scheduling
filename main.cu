#include <private/common.h>
#include <private/csv.h>
#include <private/job.h>
#include <private/machine.h>
#include <private/population.h>
#include <iostream>
#include <string>
#include "include/job_base.h"


using namespace std;

string dirName(string path);

void dataPreprocessing(job_t **_jobs, machine_t ** _machines, process_time_t *** _process_times, csv_t wip, csv_t machine_csv, csv_t eqp_recipe);

int main(int argc, const char *argv[])
{
    string path;
    if (argc < 2) {
        path = "";
    } else {
        path = argv[1];
        path = dirName(path);
    }

    csv_t wip(path + "wip.csv", "r", true, true);
    csv_t machine_csv(path + "tools.csv", "r", true, true);
    csv_t eqp_recipe(path + "recipe.csv", "r", true, true);
    int AMOUNT_OF_JOBS = wip.nrows();
    int AMOUNT_OF_MACHINES = machine_csv.nrows();
    
    job_t * jobs;
    machine_t * machines;
    process_time_t ** process_times;
    
    dataPreprocessing(&jobs, &machines, &process_times, wip, machine_csv, eqp_recipe);

    population_t pop = population_t{
        .population_number = 0,
        .parameters = {
            .AMOUNT_OF_JOBS = AMOUNT_OF_JOBS,
            .AMOUNT_OF_MACHINES = AMOUNT_OF_MACHINES,
            .AMOUNT_OF_CHROMOSOMES = 100,
            .EVOLUTION_RATE = 0.2,
            .SELECTION_RATE = 0.3
        },
        .sample = {
            .jobs = jobs,
            .machines = machines,
            .process_times = process_times
        }
    }; 

    initPopulation(&pop);

    return 0;
}


void dataPreprocessing(job_t **_jobs, machine_t ** _machines, process_time_t *** _process_times, csv_t wip, csv_t machine_csv, csv_t eqp_recipe){
    unsigned int AMOUNT_OF_JOBS = wip.nrows();
    unsigned int AMOUNT_OF_MACHINES = machine_csv.nrows();

    job_t *jobs = (job_t *) malloc(sizeof(job_t) * AMOUNT_OF_JOBS);
    machine_t *machines =
        (machine_t *) malloc(sizeof(machine_t) * AMOUNT_OF_MACHINES);
    process_time_t **process_times =
        (process_time_t **) malloc(sizeof(process_time_t *) * AMOUNT_OF_JOBS);

    for (unsigned int i = 0; i < AMOUNT_OF_JOBS; ++i) {
        createJob(&jobs[i], wip.getElements(i));
    }
    for (unsigned int i = 0; i < AMOUNT_OF_MACHINES; ++i) {
        createMachine(&machines[i], machine_csv.getElements(i));
    }
    csv_t process_time_csv;
    for (unsigned int i = 0; i < AMOUNT_OF_JOBS; ++i) {
        process_time_csv =
            eqp_recipe.filter("RECIPE", jobs[i].recipe.str_recipe);
        unsigned int nrows = process_time_csv.nrows();
        process_times[i] =
            (process_time_t *) malloc(sizeof(process_time_t) * nrows);
        for (unsigned int j = 0; j < nrows; ++j) {
            map<string, string> elements = process_time_csv.getElements(j);
            unsigned int machine_no = atoi(&elements["EQP_ID"].c_str()[3]);
            process_times[i][j] =
                process_time_t{.machine_no = machine_no,
                               .process_time = jobs[i].base.qty / 25.0 *
                                               stof(elements["PROCESS_TIME"]),
                               .ptr_derived_object = NULL};
        }
        jobs[i].base.size_of_process_time = nrows;
    }

    *_jobs = jobs;
    *_machines = machines;
    *_process_times = process_times;
}

string dirName(string path)
{
    return (path.back() == '/' ? path : path + '/');
}
