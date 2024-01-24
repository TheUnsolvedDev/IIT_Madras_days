#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define MAX_THREADS 512
#define MAX_STR_LENGTH 1024
#define MAX_STRS 20

int main()
{

    char strs[MAX_STRS][MAX_STR_LENGTH] = {
        "01234567890",
        "01234567891",
        "01234567892",
        "01234567893",
        "01234567894",
        "01234567895",
        "01234567896",
        "01234567897",
        "01234567898",
        "01234567899"};

    return 0;
}