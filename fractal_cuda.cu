/*
k vhjjbv bhjmkjhhjikkih vhjiin jioljnb ioplkb vbkbv chjp[cguX4st8,.;'
Fractal code for CS 4380 / CS 5351

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
#include <cmath>
#include <algorithm>
#include <sys/time.h>
#include <cuda.h>
#include "BMP43805351.h"

static const int ThreadsPerBlock = 1024;

static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

static __global__ void fractal(const int width, const int frames, unsigned char* const pic)
{
  const long i = threadIdx.x + blockIdx.x * (long)blockDim.x;
  if(i > width * width * frames)
  {
    return;
  }

  const double Delta = 0.00304;
  const double xMid = -0.055846456;
  const double yMid = -0.668311119;

  // compute pixels of each frame
  const int frame = i / (width * width);//for loop -columns
  double delta = Delta * pow(0.975, frame);
  //for loop -frames
  const int col = i % width;
  const double xMin = xMid - delta;
  const double yMin = yMid - delta;
  const double dw = 2.0 * delta / width;
  const int row = (i / width) % width;//for loop -rows
  const double cy = yMin + row * dw;
  const double cx = xMin + col * dw;
  double x = cx;
  double y = cy;
  double x2, y2;
  int count = 256;
  do {
    x2 = x * x;
    y2 = y * y;
    y = 2.0 * x * y + cy;
    x = x2 - y2 + cx;
    count--;
  } while ((count > 0) && ((x2 + y2) <= 5.0));
  pic[frame * width * width + row * width + col] = (unsigned char)count;
}


int main(int argc, char *argv[])
{
  printf("Fractal v2.1\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "USAGE: %s frame_width number_of_frames\n", argv[0]); exit(-1);}
  const int width = atoi(argv[1]);
  if (width < 8) {fprintf(stderr, "ERROR: frame_width must be at least 8\n"); exit(-1);}
  const int frames = atoi(argv[2]);
  if (frames < 1) {fprintf(stderr, "ERROR: number_of_frames must be at least 1\n"); exit(-1);}
  printf("frames: %d\n", frames);
  printf("width: %d\n", width);

  // allocate picture array
  unsigned char* pic = new unsigned char [frames * width * width];


  unsigned char* const picGPU = new unsigned char [frames * width * width];
  if (cudaSuccess != cudaMalloc((void **)&picGPU, sizeof(unsigned char) * frames * width * width)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  CheckCuda();

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // execute timed code
  fractal<<<((frames*width*width) + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(width, frames, picGPU);

//  CheckCuda();
  cudaDeviceSynchronize();
  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);

   CheckCuda();
   const int copySize = sizeof(unsigned char) * frames * width * width;
   if (cudaSuccess != cudaMemcpy(pic, picGPU, copySize, cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}
  // write result to BMP files
  if ((width <= 256) && (frames <= 64)) {
    for (int frame = 0; frame < frames; frame++) {
      BMP24 bmp(0, 0, width, width);
      for (int y = 0; y < width; y++) {
        for (int x = 0; x < width; x++) {
          bmp.dot(x, y, pic[frame * width * width + y * width + x] * 0x010101);
        }
      }
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      bmp.save(name);
    }
  }

  // clean up
  cudaFree(picGPU);
  delete [] pic;
  return 0;
}
