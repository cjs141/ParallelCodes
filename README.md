# ParallelCodes
### MPI, PThread, OpenMP Implementations for Computation Heavy Codes
* MPI Codes are used on distributed network systems
* PThread and OpenMP codes are used on shared memory systems


## MPI Code Example:
Collatz Sequencing
```
#include <cstdio>
#include <algorithm>
#include <sys/time.h>
#include <mpi.h>

static int collatz(const long bound, int my_rank, int comm_sz)
{
  int my_maxlen = 0;
  //Cyclic partion
  int i;
  for(i = my_rank + 1; i <= bound; i += comm_sz)//cyclic distribution
  {
    long val = i;
    int len = 1;
    while (val != 1) {
      len++;
      if ((val % 2) == 0) {
        val /= 2;  // even
      } else {
        val = 3 * val + 1;  // odd
      }
    }
    my_maxlen = std::max(my_maxlen, len);
  }
  return my_maxlen;
}

int main(int argc, char *argv[])
{
  int maxlen;
  //initialize the MPI library
  int comm_sz, my_rank;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int *res = NULL;

  if(my_rank == 0) printf("Collatz v1.4\n");

  // check command line
  if(argc != 2) {fprintf(stderr, "USAGE: %s upper_bound\n", argv[0]); exit(-1);}
  const long bound = atol(argv[1]);
  if(bound < 1) {fprintf(stderr, "ERROR: upper_bound must be at least 1\n"); exit(-1);}
  if(my_rank == 0) {
     printf("upper bound: %ld\n", bound);
     printf("processes: %d\n", comm_sz);
  }

  // start time
  timeval start, end;
  MPI_Barrier(MPI_COMM_WORLD); // for better timing
  gettimeofday(&start, NULL);

  // execute timed code
  const int my_maxlen = collatz(bound, my_rank, comm_sz);

  MPI_Reduce(&my_maxlen, &maxlen, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);//Find max and store in root

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  if(my_rank == 0) printf("compute time: %.6f s\n", runtime);

  // print result
  if(my_rank == 0) printf("longest sequence length: %d elements\n", maxlen);



  // clean up
  MPI_Finalize();

  return 0;
}
```
## PThread Code Example:
Collatz Sequencing
 ```
 #include <cstdio>
#include <algorithm>
#include <sys/time.h>
#include <pthread.h>

//shared variables
static int maxlen;
static long bound;
static long threads;
pthread_mutex_t lock;



static void* collatz(void* arg)
{
  const long my_rank = (long)arg;
  // compute sequence lengths
  int my_maxlen = 0;
  //cyclic partition
  int i;
  for(i = my_rank + 1; i <= bound; i += threads) {
    long val = i;
    int len = 1;
    while (val != 1) {
      len++;
      if ((val % 2) == 0) {
        val /= 2;  // even
      } else {
        val = 3 * val + 1;  // odd
      }
    }
    my_maxlen = std::max(my_maxlen, len);
  }

  pthread_mutex_lock(&lock);
  maxlen = std::max(maxlen, my_maxlen);
  pthread_mutex_unlock(&lock);
  return NULL;
}

int main(int argc, char *argv[])
{
  printf("Collatz v1.4\n");

  // check command line
  bound = atol(argv[1]);
  if (argc != 3) {fprintf(stderr, "USAGE: %s upper_bound\n", argv[0]); exit(-1);}
  if (bound < 1) {fprintf(stderr, "ERROR: upper_bound must be at least 1\n"); exit(-1);}
  printf("upper bound: %ld\n", bound);
  threads = atol(argv[2]);
  if (threads < 1) {fprintf(stderr, "ERROR: threads must be at least 1\n"); exit(-1);}
  printf("threads: %ld\n", threads);

  if(pthread_mutex_init(&lock, NULL) != 0) {
        printf("\n mutex init has failed\n");
        return 1;
    }

  // initialize pthread variables
  pthread_t* const handle = new pthread_t [threads - 1];

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // launch threads
  for (long thread = 0; thread < threads - 1; thread++) {
    pthread_create(&handle[thread], NULL, collatz, (void *)thread);
  }

  // execute timed code
  collatz((void*)(threads - 1));



  // join threads
  for (long thread = 0; thread < threads - 1; thread++) {
    pthread_join(handle[thread], NULL);
  }
  pthread_mutex_destroy(&lock);
  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);

  // print result
  printf("longest sequence length: %d elements\n", maxlen);
  return 0;
}
```
