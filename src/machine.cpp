#include <private/machine.h>

void createMachine(machine_t * machine, std::map<std::string, std::string> elements){
    machine->base.avaliable_time = std::stoi(elements["RECOVER_TIME"]);
    machine->base.machine_no = atoi(&elements["EQP_ID"].c_str()[3]);

}
