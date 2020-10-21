/*
collatz code for CS 4380 / CS 5351

Copyright (c) 2020 Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is *not* permitted. Use in source or binary form, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

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

