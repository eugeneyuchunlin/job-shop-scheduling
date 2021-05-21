#include <include/common.h>

#define greater(a, b, type) *(type *) a > *(type *) b

int cmpint(const void *a, const void *b)
{
    return greater(a, b, int);
}

int cmpdouble(const void *a, const void *b)
{
    return greater(a, b, double);
}
