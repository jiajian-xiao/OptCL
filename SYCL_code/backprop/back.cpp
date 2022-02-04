#include <CL/sycl.hpp>

#include <array>
#include <iostream>
#include <float.h>

#define PI 3.141592653589793238463
#define e 2.718281828459045

#define epsilon 0.05

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
constexpr cl::sycl::access::mode sycl_read_write = cl::sycl::access::mode::read_write;

class kernel1;

class kernel2;

class kernel3;

float sigmoid(float x) {
  return (1.0f / (1.0f + cl::sycl::exp(-x)));
}

float f_theta(float x, float b, float* C, float* W, float* V, size_t N) {
  float result = b;
  for (size_t i = 0; i < N; i++) {
    result += V[i] * sigmoid(C[i] + W[i] * x);
  }
  return result;
}

void train(float X, float Y, float *b, float* C, float* W, float* V, size_t N) {
  cl::sycl::cpu_selector device_selector;
  cl::sycl::queue deviceQueue(device_selector);

  float B = *b;

  cl::sycl::range<1> numOfN{N};
  
  cl::sycl::buffer<float, 1> bufferC(C, numOfN);
  cl::sycl::buffer<float, 1> bufferW(W, numOfN);
  cl::sycl::buffer<float, 1> bufferV(V, numOfN);

  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorC = bufferC.get_access<sycl_read>(cgh);
    auto accessorW = bufferW.get_access<sycl_write>(cgh);
    auto accessorV = bufferV.get_access<sycl_read>(cgh);

    auto kern1 = [=](cl::sycl::id<1> id) {
        accessorW[id] = accessorW[id] - 0.05 * 2 * (f_theta(X, B, accessorC.get_pointer(), accessorW.get_pointer(), accessorV.get_pointer(), N) - Y) * accessorV[id] * X * 
               (1 - sigmoid(accessorC[id] + accessorW[id] * X)) * sigmoid(accessorC[id] + accessorW[id] * X);
    };
    cgh.parallel_for<class kernel1>(numOfN, kern1);
  });



  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorC = bufferC.get_access<sycl_read>(cgh);
    auto accessorW = bufferW.get_access<sycl_read>(cgh);
    auto accessorV = bufferV.get_access<sycl_write>(cgh);
    // auto accessorB = bufferB.get_access<sycl_read>(cgh);

    auto kern2 = [=](cl::sycl::id<1> id) {
        accessorV[id] = accessorV[id] - 0.05 * 2 * (f_theta(X, B, accessorC.get_pointer(), accessorW.get_pointer(), accessorV.get_pointer(), N) - Y) * sigmoid(accessorC[id] + accessorW[id] * X);
    };
    cgh.parallel_for<class kernel2>(numOfN, kern2);
  });

  auto pointsC = bufferC.get_access<sycl_read>().get_pointer();
  auto pointsW = bufferW.get_access<sycl_read>().get_pointer();
  auto pointsV = bufferV.get_access<sycl_read>().get_pointer();
  B = B - 0.05 * 2 * (f_theta(X, B, pointsC, pointsW, pointsV, N) - Y);

  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorC = bufferC.get_access<sycl_write>(cgh);
    auto accessorW = bufferW.get_access<sycl_read>(cgh);
    auto accessorV = bufferV.get_access<sycl_read>(cgh);

    auto kern3 = [=](cl::sycl::id<1> id) {
        accessorC[id] = accessorC[id] - 0.05 * 2 * (f_theta(X, B, accessorC.get_pointer(), accessorW.get_pointer(), accessorV.get_pointer(), N) - Y) * accessorV[id] * 
               (1 - sigmoid(accessorC[id] + accessorW[id] * X)) * sigmoid(accessorC[id] + accessorW[id] * X);
    };
    cgh.parallel_for<class kernel3>(numOfN, kern3);
  });

  *b = B;
}

int main(int argc, char** argv) {
  std::string numOfN = argv[1];

  size_t N = stoi(numOfN);
  size_t train_size = 512;
  size_t epoch = 5;

  std::cout << "N: " << N << std::endl;
  
  float* C = (float*)malloc(sizeof(float) * N);
  float* W = (float*)malloc(sizeof(float) * N);
  float* V = (float*)malloc(sizeof(float) * N);
  float B = 0.0;

  float* train_X = (float*)malloc(sizeof(float) * train_size);
  float* train_Y = (float*)malloc(sizeof(float) * train_size);


  for (size_t i = 0; i < N; i++) {
    W[i] = 2 * 0.3 -1.f;
    V[i] = 2 * 0.3 -1.f;
    C[i] = 2 * 0.3 -1.f;
    // printf("%f %f %f\n", W[i],V[i],C[i] );
  }


  for (size_t i = 0; i < train_size; i++) {
    train_X[i] = (float)i * 2.f * PI / (float)train_size,
    train_Y[i] = sin((float)i * 2.f * PI / (float)train_size);
  }


  auto start = std::chrono::steady_clock::now();
  for (size_t j = 0; j < epoch; j++) {
    for (size_t i = 0; i < train_size; i++) {
      train(train_X[i], train_Y[i], &B, C, W, V, N);
    }
  }
  auto end = std::chrono::steady_clock::now();

  auto time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();


  std::cout << "Time: " << time << " milliseconds" << std::endl;

  return 0;
}
