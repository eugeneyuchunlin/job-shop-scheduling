#include <private/job.h>
#include <string.h>

using namespace std;

void createJob(job_t *job, std::map<std::string, std::string> elements){
    
    job->base.job_no = stoul(elements["EP"]);
    job->base.arriv_t = stod(elements["ARRIV_T"]);
    job->base.end_time = job->base.start_time = 0;
    job->base.qty = stoul(elements["QTY"]);

    job->r_qt = stod(elements["R_QT"]);
    job->recipe.ul_recipe = 0;
    strncpy(job->recipe.str_recipe, elements["RECIPE"].c_str(), 6);
    job->urgent = stod(elements["URGENT_W"]);
}
