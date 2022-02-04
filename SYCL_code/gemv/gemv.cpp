#include <CL/sycl.hpp>

#include <array>
#include <iostream>

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

/* floathis is the class used to name the kernel for the runtime.
 * floathis must be done when the kernel is expressed as a lambda. */
class MVM1;

class MVM2;

class VVA;

void gemv(float* MA, float* MB, float* MC, float* MD, float* ME, float* MF, float* MG, size_t N, size_t M) {
  cl::sycl::gpu_selector device_selector;
  cl::sycl::queue deviceQueue(device_selector);

  cl::sycl::range<1> numOfN{N};
  cl::sycl::range<1> numOfM{M};
  cl::sycl::range<1> sizeofMatrix{N*M};

  cl::sycl::buffer<float, 1> bufferA(MA, sizeofMatrix);
  cl::sycl::buffer<float, 1> bufferB(MB, numOfN);
  cl::sycl::buffer<float, 1> bufferC(MC, numOfM);

  cl::sycl::buffer<float, 1> bufferD(MD, sizeofMatrix);
  cl::sycl::buffer<float, 1> bufferE(ME, numOfN);
  cl::sycl::buffer<float, 1> bufferF(MF, numOfM);

  cl::sycl::buffer<float, 1> bufferG(MG, numOfM);

  // cl::sycl::buffer<float, 1> bufferD(VD.data(), numOfItems);

  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);
    
    auto kern1 = [=](cl::sycl::id<1> wiID) {
      float sum = 0.f;
      for (size_t j = 0; j < N; j++) {
        sum += accessorA[N*wiID+j] * accessorB[j];
      }
      accessorC[wiID] = sum;
    };
    cgh.parallel_for<class MVM1>(numOfM, kern1);
  });

  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorD = bufferD.template get_access<sycl_read>(cgh);
    auto accessorE = bufferE.template get_access<sycl_read>(cgh);
    auto accessorF = bufferF.template get_access<sycl_write>(cgh);

    auto kern2 = [=](cl::sycl::id<1> wiID) {
        float sum1 = 0.f;
        for (size_t j = 0; j < N; j++) {
          sum1 += accessorD[N*wiID+j] * accessorE[j];
        }
        accessorF[wiID] = sum1;
    };
    cgh.parallel_for<class MVM2>(numOfM, kern2);
  });

  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorC = bufferC.template get_access<sycl_read>(cgh);
    auto accessorF = bufferF.template get_access<sycl_read>(cgh);
    auto accessorG = bufferG.template get_access<sycl_write>(cgh);

    auto kern3 = [=](cl::sycl::id<1> wiID) {
      accessorG[wiID] = accessorC[wiID] + accessorF[wiID];
    };
    cgh.parallel_for<class VVA>(numOfM, kern3);
  });
}

int main(int argc, char** argv) {
  size_t N = 1024;
  size_t M = 256;
  std::string numOfN= argv[1];
  std::string numOfM= argv[2];
  N = stoi(numOfN);
  M = stoi(numOfM);

  std::cout << "N x M : " << N << " x " << M << std::endl;

  float* MA;
  float* MB;
  float* MC;
  float* MD;
  float* ME;
  float* MF;

  float* MG;

  MA = new float[N * M];
  MB = new float[N];
  MC = new float[M];

  MD = new float[N * M];
  ME = new float[N];
  MF = new float[M];

  MG = new float[M];

  for (size_t k=0;k<N*M;k++){
      MA[k] = ((float)rand()/(float)RAND_MAX)*10.;
  }
  for (size_t k=0;k<N;k++){
      MB[k] = ((float)rand()/(float)RAND_MAX)*10.;
  }
  for (size_t k=0;k<N*M;k++){
      MD[k] = ((float)rand()/(float)RAND_MAX)*10.;
  }
  for (size_t k=0;k<N;k++){
      ME[k] = ((float)rand()/(float)RAND_MAX)*10.;
  }


  auto start = std::chrono::steady_clock::now();
  for (size_t iter = 0 ; iter < 10 ; iter++) {
    gemv(MA, MB, MC, MD, ME, MF, MG, N, M);
  }
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();


  std::cout << "time: " << time << " milliseconds" << std::endl;

  return 0;
}
