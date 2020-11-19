/*
Ray-tracing code for CS 4380 / CS 5351

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

Author: Martin Burtscher (idea from Ronald Rahaman)
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

static void prepare(int frames, int width, long* ball_y)
{
    const int semiwidth = width / 2;
    ball_y[0] = semiwidth;
    float vel = 0.0f;

    for(int i = 1; i < frames; i++)
    {
        ball_y[i] = ball_y[i-1] + vel;
        vel -= width * 0.0005f;  // acceleration
        vel *= 0.998f; // dampening
        if (ball_y[i] < -semiwidth) {
            ball_y[i] = -width - ball_y[i];
            vel = -vel;
        }
    }
}

static __global__ void raytrace(const int width, const int frames, unsigned char* const pic, long* ball_y)
{

  const long i = threadIdx.x + blockIdx.x * (long)blockDim.x;
  if(i > width * width * frames)
  {
    return;
  }
  const int semiwidth = width / 2;
  const int pix_x = (i % width) - semiwidth;
  const int pix_y = ((i / width) % width) - semiwidth;
  const int frame = i / (width * width);

  // eye is at <0, 0, 0>

  // initialize ball
  const float ball_r = semiwidth / 3;  // radius of ball
  float ball_z = semiwidth * 3;

  // initialize light source
  const float sol_x = semiwidth * -64;
  const float sol_y = semiwidth * 64;
  const float sol_z = semiwidth * -16;

  // compute pixels of each frame
  float ball_x = frame * width * 0.004f - semiwidth;

  // send one ray through each pixel
  const float c = ball_x * ball_x + ball_y[frame] * ball_y[frame] + ball_z * ball_z - ball_r * ball_r;
  const int pix_z = semiwidth * 2;
  const int a = pix_x * pix_x + pix_y * pix_y + pix_z * pix_z;
  const float e = pix_x * ball_x + pix_y * ball_y[frame] + pix_z * ball_z;
  const float d = e * e - a * c;
  if (d >= 0.0f) {  // ray hits ball
    const float ds = sqrtf(d);
    const float k1 = (e + ds) / a;
    const float k2 = (e - ds) / a;
    const float k3 = fminf(k1, k2);
    const float k4 = fmaxf(k1, k2);
    if (k4 > 0.0f) {  // in front of (not behind) eye
      const float k = (k3 > 0.0f) ? k3 : k4;

      // ball surface normal at loc where ray hits
      const float n_x = k * pix_x - ball_x;
      const float n_y = k * pix_y - ball_y[frame];
      const float n_z = k * pix_z - ball_z;

      // vector to light source from point where ray hits
      const float s_x = sol_x - k * pix_x;
      const float s_y = sol_y - k * pix_y;
      const float s_z = sol_z - k * pix_z;

      // cosine between two vectors
      const float p = s_x * n_x + s_y * n_y + s_z * n_z;
      const float ls = sqrtf(s_x * s_x + s_y * s_y + s_z * s_z);
      const float ln = sqrtf(n_x * n_x + n_y * n_y + n_z * n_z);
      const float cos = p / (ls * ln);

      if (cos > 0) {  // is lit by light source
        const unsigned char brightness = cos * 255.0f;
	pic[frame * width * width + (pix_y + semiwidth) * width + (pix_x + semiwidth)] = brightness;
      }
    }
  }
}


int main(int argc, char *argv[])
{
  printf("Ray Tracing v1.0\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "USAGE: %s frame_width number_of_frames\n", argv[0]); exit(-1);}
  int width = atoi(argv[1]);
  if (width < 100) {fprintf(stderr, "ERROR: frame_width must be at least 100\n"); exit(-1);}
  if ((width % 2) != 0) {fprintf(stderr, "ERROR: frame_width must be even\n"); exit(-1);}
  int frames = atoi(argv[2]);
  if (frames < 1) {fprintf(stderr, "ERROR: number_of_frames must be at least 1\n"); exit(-1);}
  printf("frames: %d\n", frames);
  printf("width: %d\n", width);

  // allocate picture array
  unsigned char* pic = new unsigned char [frames * width * width];
  long* ball_y = new long[frames];
  long* ball_yGPU = new long[frames];
  unsigned char* const picGPU = new unsigned char [frames * width * width];
  if (cudaSuccess != cudaMalloc((void **)&picGPU, sizeof(unsigned char) * frames * width * width)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
//  CheckCuda();
  if (cudaSuccess != cudaMalloc((void **)&ball_yGPU, sizeof(long) * frames)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
//  CheckCuda();


  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  prepare(frames, width, ball_y);
  if (cudaSuccess != cudaMemcpy(ball_yGPU, ball_y, sizeof(long) * frames, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}
// CheckCuda();
 // execute timed code
  raytrace<<<((frames*width*width) + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(width, frames, picGPU, ball_yGPU);
//CheckCuda();
// end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);
  CheckCuda();
  // write result to BMP files
  if ((width <= 256) && (frames <= 80)) {
    for (int frame = 0; frame < frames; frame++) {
      BMP24 bmp(0, 0, width, width);
      for (int y = 0; y < width; y++) {
        for (int x = 0; x < width; x++) {
          bmp.dot(x, y, pic[frame * width * width + y * width + x] * 0x010101);
        }
      }
      char name[32];
      sprintf(name, "raytrace%d.bmp", frame + 1000);
      bmp.save(name);
    }
  }

  // clean up
  cudaFree(picGPU);
  cudaFree(ball_yGPU);
  delete [] pic;
  return 0;
}
