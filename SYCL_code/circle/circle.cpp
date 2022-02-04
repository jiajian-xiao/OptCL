#include <CL/sycl.hpp>

#include <array>
#include <iostream>
#include <float.h>


constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
constexpr cl::sycl::access::mode sycl_read_write = cl::sycl::access::mode::read_write;

class kernel1;


struct Point {
    float x, y;     // coordinates

    Point() :
        x(.0),
        y(.0){}

    Point(float x, float y) :
        x(x),
        y(y){}
};

void circle(Point* points, size_t Len) {
  cl::sycl::gpu_selector device_selector;
  cl::sycl::queue deviceQueue(device_selector);

  cl::sycl::range<1> numOfLen{Len};
  cl::sycl::buffer<Point, 1> bufferPoint(points, numOfLen);

  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorPoint = bufferPoint.get_access<sycl_read_write>(cgh);
   
    auto kern1 = [=](cl::sycl::id<1> id) {
      Point p = accessorPoint[id];
      Point new_p;
      new_p.x = p.x ;
      new_p.y = p.y ;
      for (size_t j = 0; j < Len; j++)
      {
        Point c = accessorPoint[j];
        float dist = cl::sycl::sqrt((p.x - c.x) * (p.x - c.x) + (p.y - c.y) * (p.y - c.y));
        if (dist < 20.0) {
          float sep_dist = dist - 10.0;
          Point force;
          if (sep_dist > 0.0) {
            force.x = 0.05 * (20.0 - dist) * (c.x - p.x) / dist;
            force.y = 0.05 * (20.0 - dist) * (c.y - p.y) / dist;
          } else {
            force.x = 0.1 * (c.x - p.x);
            force.y = 0.1 * (c.y - p.y);
          }          
          new_p.x += force.x;
          new_p.y += force.y;
        }
      }
      accessorPoint[id] = new_p;
    };
    cgh.parallel_for<class kernel1>(numOfLen, kern1);
  });
}

int main(int argc, char** argv) {
  size_t Width;
  size_t Length;
  size_t iteration = 100;
  std::string numOfPoints = argv[1];
  size_t len = stoi(numOfPoints);

  Width = 128;//sqrt((float)len/0.05);
  Length = 128;//sqrt((float)len/0.05);

  std::cout << "Number of points " << len << " Length: " << Length << " Width:" << Width << std::endl;
  Point* points;
  points = new Point[len];

  for (size_t i=0;i<len;i++) {
    float X = (float)rand()/(float)RAND_MAX*(float)Width;
    float Y = (float)rand()/(float)RAND_MAX*(float)Length;
    Point p1 = Point(X, Y);
    points[i] = p1;
  }

  auto start = std::chrono::steady_clock::now();
  for (size_t iter = 0 ; iter < iteration ; iter++) {
    circle(points, len);
  }
  auto end = std::chrono::steady_clock::now();

  auto time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  std::cout << "Time: " << time << " milliseconds" << std::endl;

  return 0;
}
