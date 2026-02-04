#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <limits.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, sizeof(hostname));
    hostname[sizeof(hostname)-1] = '\0';

    printf("Hello MPI! rank=%d/%d host=%s\n", rank, size, hostname);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}
