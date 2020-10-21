/*
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

