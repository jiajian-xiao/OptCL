#include <CL/sycl.hpp>

#include <array>
#include <iostream>
#include <float.h>


constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
constexpr cl::sycl::access::mode sycl_read_write = cl::sycl::access::mode::read_write;

class kernel1;

class kernel2;


void srad(float* image, size_t Nr, size_t Nc, size_t Ne, float* dN, float* dS, float* dW, float* dE, float* c, float q0sqr) {
  cl::sycl::gpu_selector device_selector;
  cl::sycl::queue deviceQueue(device_selector);

  cl::sycl::range<1> numOfNe{Ne};

  cl::sycl::buffer<float, 1> bufferImage(image, numOfNe);
  cl::sycl::buffer<float, 1> bufferdN(dN, numOfNe);
  cl::sycl::buffer<float, 1> bufferdS(dS, numOfNe);
  cl::sycl::buffer<float, 1> bufferdW(dW, numOfNe);
  cl::sycl::buffer<float, 1> bufferdE(dE, numOfNe);
  cl::sycl::buffer<float, 1> bufferC(c, numOfNe);


  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorImage = bufferImage.get_access<sycl_read>(cgh);
    auto accessordN = bufferdN.get_access<sycl_write>(cgh);
    auto accessordS = bufferdS.get_access<sycl_write>(cgh);
    auto accessordW = bufferdW.get_access<sycl_write>(cgh);
    auto accessordE = bufferdE.get_access<sycl_write>(cgh);
    auto accessorC = bufferC.get_access<sycl_write>(cgh);

    auto kern1 = [=](cl::sycl::id<1> id) {
      float Jc = accessorImage[id];
      float ddN, ddS, ddW, ddE;
      int index = static_cast<int>(id[0]);

      accessordN[id] = (index % Nr == 0)?accessorImage[id]:accessorImage[id-1] - Jc;                
      accessordS[id] = ((index+1) % Nr ==0)?accessorImage[id]:accessorImage[id+1] - Jc;               
      accessordW[id] = (index < Nr)?accessorImage[id]:accessorImage[id-Nr] - Jc;             
      accessordE[id] = (index > Nr*(Nc-1)-1)?accessorImage[id]:accessorImage[id+Nr] - Jc;
      float G2 = (accessordN[id]*accessordN[id] + accessordS[id]*accessordS[id] + accessordW[id]*accessordW[id] + accessordE[id]*accessordE[id]) / (Jc*Jc);
      // normalized discrete laplacian (equ 54)
      float L = (accessordN[id] + accessordS[id] + accessordW[id] + accessordE[id]) / Jc;         // laplacian (based on derivatives)

      // ICOV (equ 31/35)
      float num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;            // num (based on gradient and laplacian)
      float den  = 1.0 + (.25*L);                     // den (based on laplacian)
      float qsqr = num/(den*den);                   // qsqr (based on num and den)

      // diffusion coefficent (equ 33) (every element of IMAGE)
      den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;          // den (based on qsqr and q0sqr)
      accessorC[id] = 1.0 / (1.0+den) ;                  // diffusion coefficient (based on den)

      // saturate diffusion coefficent to 0-1 range
      if (accessorC[id] < 0)                       // if diffusion coefficient < 0
        accessorC[id] = 0;                       // ... set to 0
      else if (accessorC[id] > 1)                      // if diffusion coefficient > 1
        accessorC[id] = 1; 
    };
    cgh.parallel_for<class kernel1>(numOfNe, kern1);
  });

  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorImage = bufferImage.get_access<sycl_write>(cgh);
    auto accessordN = bufferdN.get_access<sycl_read>(cgh);
    auto accessordS = bufferdS.get_access<sycl_read>(cgh);
    auto accessordW = bufferdW.get_access<sycl_read>(cgh);
    auto accessordE = bufferdE.get_access<sycl_read>(cgh);
    auto accessorC = bufferC.get_access<sycl_read>(cgh);

    auto kern2 = [=](cl::sycl::id<1> id) {
      float ccS, ccE;
      int index = static_cast<int>(id[0]);

      float cN = accessorC[id];                          // north diffusion coefficient
      float cS = ((index+1) % Nr ==0)?accessorC[id]:accessorC[id+1];                   // south diffusion coefficient
      float cW = accessorC[id];                          // west diffusion coefficient
      float cE = (index > Nr*(Nc-1)-1)?accessorC[id]:accessorC[id+Nr]; 

      float D = cN*accessordN[id] + cS*accessordS[id] + cW*accessordW[id] + cE*accessordE[id];       // divergence

      // image update (equ 61) (every element of IMAGE)
      accessorImage[id] = accessorImage[id] + 0.25*1.0*D;      
    };
    cgh.parallel_for<class kernel2>(numOfNe, kern2);
  });



}

int main(int argc, char** argv) {
  std::string numOfNr = argv[1];
  std::string numOfNc = argv[2];

  size_t Nr = stoi(numOfNr);
  size_t Nc = stoi(numOfNc);

  std::cout << "Nr x Nc " << Nr << " x " << Nc << std::endl;

  size_t Ne = Nr*Nc;
  size_t epoch = 100;

  float* image = (float*)malloc(sizeof(float) * Ne);

  for (size_t p = 0 ; p < Ne ; p++){
    image[p] = (float)p/(float)Ne+1;
    // std::cout << p << " " << image[p] << std::endl;
  }

  float* dN = (float*)malloc(sizeof(float)*Ne) ;                      // north direction derivative
  float* dS = (float*)malloc(sizeof(float)*Ne) ;                      // south direction derivative
  float* dW = (float*)malloc(sizeof(float)*Ne) ;                      // west direction derivative
  float* dE = (float*)malloc(sizeof(float)*Ne) ;                      // east direction derivative

  float* c  = (float*)malloc(sizeof(float)*Ne) ;                      

  int r1,r2,c1,c2;
  long NeROI; 

  r1     = 0;                   
  r2     = Nr - 1;                 
  c1     = 0;                     
  c2     = Nc - 1;          
  NeROI = (r2-r1+1)*(c2-c1+1); 

  float tmp, sum, sum2;
  float meanROI, varROI, q0sqr;   

  auto start = std::chrono::steady_clock::now();
  for (size_t iter=0;iter<epoch;iter++) {
    sum=0;
    sum2=0;
    for (int i=r1; i<=r2; i++) {                     
      for (int j=c1; j<=c2; j++) {                   
          tmp   = image[i + Nr*j];                   
          sum  += tmp ;                     
          sum2 += tmp*tmp;                      
      }
    }
    meanROI = sum / NeROI;                        
    varROI  = (sum2 / NeROI) - meanROI*meanROI;             
    q0sqr   = varROI / (meanROI*meanROI);   

    srad(image, Nr, Nc, Ne, dN, dS, dW, dE, c, q0sqr);
  }
  auto end = std::chrono::steady_clock::now();

  auto time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  std::cout << "Time: " << time << " milliseconds" << std::endl;

  return 0;
}
