#include <CL/sycl.hpp>

#include <array>
#include <iostream>
#include <float.h>

#define MAX_PD  (3.0e6)
/* required precision in degrees  */
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor */
#define FACTOR_CHIP 0.5

/* chip parameters  */
float t_chip = 0.0005;
float amb_temp = 80.0;

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
constexpr cl::sycl::access::mode sycl_read_write = cl::sycl::access::mode::read_write;

class kernel1;


void single_iteration(float *temp, float *power, size_t row, size_t col,
            float Cap, float Rx, float Ry, float Rz, 
            float step) {
  cl::sycl::cpu_selector device_selector;
  cl::sycl::queue deviceQueue(device_selector);

  cl::sycl::range<1> numOfN{col*row};


  cl::sycl::buffer<float, 1> bufferPower(power, numOfN);
  cl::sycl::buffer<float, 1> bufferTemp(temp, numOfN);


  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorPower = bufferPower.get_access<sycl_read>(cgh);
    auto accessorTemp = bufferTemp.get_access<sycl_read_write>(cgh);

    auto kern1 = [=](cl::sycl::id<1> id) {
      int c = id[0] % col; 
      int r = id[0] / row; 
      
      /*  Corner 1  */
      float delta = 0.0;
      if ( (r == 0) && (c == 0) ) {
        delta = (step / Cap) * (accessorPower[0] +
            (accessorTemp[1] - accessorTemp[0]) / Rx +
            (accessorTemp[col] - accessorTemp[0]) / Ry +
            (80.0 - accessorTemp[0]) / Rz);
      } /*  Corner 2  */
      else if ((r == 0) && (c == col-1)) {
        delta = (step / Cap) * (accessorPower[c] +
            (accessorTemp[c-1] - accessorTemp[c]) / Rx +
            (accessorTemp[c+col] - accessorTemp[c]) / Ry +
            (80.0 - accessorTemp[c]) / Rz);
      } /*  Corner 3  */
      else if ((r == row-1) && (c == col-1)) {
        delta = (step / Cap) * (accessorPower[r*col+c] + 
            (accessorTemp[r*col+c-1] - accessorTemp[r*col+c]) / Rx + 
            (accessorTemp[(r-1)*col+c] - accessorTemp[r*col+c]) / Ry + 
            (80.0 - accessorTemp[r*col+c]) / Rz);         
      } /*  Corner 4  */
      else if ((r == row-1) && (c == 0)) {
        delta = (step / Cap) * (accessorPower[r*col] + 
            (accessorTemp[r*col+1] - accessorTemp[r*col]) / Rx + 
            (accessorTemp[(r-1)*col] - accessorTemp[r*col]) / Ry + 
            (80.0 - accessorTemp[r*col]) / Rz);
      } /*  Edge 1  */
      else if (r == 0) {
        delta = (step / Cap) * (accessorPower[c] + 
            (accessorTemp[c+1] + accessorTemp[c-1] - 2.0*accessorTemp[c]) / Rx + 
            (accessorTemp[col+c] - accessorTemp[c]) / Ry + 
            (80.0 - accessorTemp[c]) / Rz);
      } /*  Edge 2  */
      else if (c == col-1) {
        delta = (step / Cap) * (accessorPower[r*col+c] + 
            (accessorTemp[(r+1)*col+c] + accessorTemp[(r-1)*col+c] - 2.0*accessorTemp[r*col+c]) / Ry + 
            (accessorTemp[r*col+c-1] - accessorTemp[r*col+c]) / Rx + 
            (80.0 - accessorTemp[r*col+c]) / Rz);
      } /*  Edge 3  */
      else if (r == row-1) {
        delta = (step / Cap) * (accessorPower[r*col+c] + 
            (accessorTemp[r*col+c+1] + accessorTemp[r*col+c-1] - 2.0*accessorTemp[r*col+c]) / Rx + 
            (accessorTemp[(r-1)*col+c] - accessorTemp[r*col+c]) / Ry + 
            (80.0 - accessorTemp[r*col+c]) / Rz);
      } /*  Edge 4  */
      else if (c == 0) {
        delta = (step / Cap) * (accessorPower[r*col] + 
            (accessorTemp[(r+1)*col] + accessorTemp[(r-1)*col] - 2.0*accessorTemp[r*col]) / Ry + 
            (accessorTemp[r*col+1] - accessorTemp[r*col]) / Rx + 
            (80.0 - accessorTemp[r*col]) / Rz);
      } /*  Inside the chip */
      else {
        delta = (step / Cap) * (accessorPower[r*col+c] + 
            (accessorTemp[(r+1)*col+c] + accessorTemp[(r-1)*col+c] - 2.0*accessorTemp[r*col+c]) / Ry + 
            (accessorTemp[r*col+c+1] + accessorTemp[r*col+c-1] - 2.0*accessorTemp[r*col+c]) / Rx + 
            (80.0 - accessorTemp[r*col+c]) / Rz);
      }
        
      /*  Update Temperatures */
      accessorTemp[r*col+c] = accessorTemp[r*col+c]+ delta;
    };
    cgh.parallel_for<class kernel1>(numOfN, kern1);
  });

}

int main(int argc, char** argv) {
  std::string numOfRow = argv[1];
  std::string numOfCol = argv[2];

  size_t row = stoi(numOfRow);
  size_t col = stoi(numOfCol);

  std::cout << "row x col " << row << " x " << col << std::endl;

  size_t iteration = 100;
  float *temp, *power;

  float chip_height = 0.16*(float)((float)row/1024);
  float chip_width = 0.16*(float)((float)col/1024);

  temp   = (float *) calloc (row * col, sizeof(float));
  power  = (float *) calloc (row * col, sizeof(float));

  for (size_t j = 0 ; j < row*col ; j++){
    power[j] = (float)((float)rand()/(RAND_MAX)); 
    temp[j] = power[j];
  }
  float grid_height = chip_height / (float)row;
  float grid_width = chip_width / (float)col;

  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
  float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
  float Rz = t_chip / (K_SI * grid_height * grid_width);

  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float step = PRECISION / max_slope;

  fprintf(stdout, "total iterations: %zu s\tstep size: %g s\n", iteration, step);
  fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iteration ; i++)
  {
    single_iteration(temp, power, row, col, Cap, Rx, Ry, Rz, step);
    for (size_t j = 0 ; j < row*col ; j++){
      power[j] = temp[j];
    }
  } 

  auto end = std::chrono::steady_clock::now();

  auto time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();


  std::cout << "Time: " << time << " milliseconds" << std::endl;

  return 0;
}
