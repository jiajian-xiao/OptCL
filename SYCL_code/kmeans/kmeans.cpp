#include <CL/sycl.hpp>
#include <array>
#include <iostream>


constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
constexpr cl::sycl::access::mode sycl_read_write = cl::sycl::access::mode::read_write;

#define FLT_MAX 2147483647.f

class kernel1;

class kernel2;

struct Point {
    float x, y;
    int cluster;
    float minDist;

    Point() : 
        x(0), 
        y(0),
        cluster(-1),
        minDist(FLT_MAX) {}
        
    Point(float x, float y) : 
        x(x), 
        y(y),
        cluster(-1),
        minDist(FLT_MAX) {}

    float distance(Point p) {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    }
};

void kmean(Point* points, Point* centroids,  size_t Len, size_t K, int *nPoints, float* sumX, float* sumY) {
  cl::sycl::gpu_selector device_selector;
  cl::sycl::queue deviceQueue(device_selector);

  cl::sycl::range<1> numOfLen{Len};
  cl::sycl::range<1> numOfK{K};

  for (size_t i=0;i<K;i++) {
    nPoints[i] = 0;
    sumX[i] = 0;
    sumY[i] = 0;
  }

  cl::sycl::buffer<Point, 1> bufferPoint(points, numOfLen);
  cl::sycl::buffer<Point, 1> bufferCentroids(centroids, numOfK);

  cl::sycl::buffer<int, 1> buffernPoints(nPoints, numOfK);
  cl::sycl::buffer<float, 1> bufferSumX(sumX, numOfK);
  cl::sycl::buffer<float, 1> bufferSumY(sumY, numOfK);

  
  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorPoint = bufferPoint.get_access<sycl_write>(cgh);
    auto accessorCentroids = bufferCentroids.get_access<sycl_read>(cgh);
   
    auto kern1 = [=](cl::sycl::id<1> id) {
      Point p = accessorPoint[id];
      for (size_t j = 0; j < K; j++){
        Point c = accessorCentroids[j];
        float dist = c.distance(p);
        if (dist < p.minDist) {
          p.minDist = dist;
          p.cluster = j;
        }
      }
      accessorPoint[id].cluster = p.cluster;
    };
    cgh.parallel_for<class kernel1>(numOfLen, kern1);
  });

  auto pointsInter = bufferPoint.get_access<sycl_read>();
  for (int i=0;i<Len;i++) {
    int clusterId = pointsInter[i].cluster;
    sumX[clusterId] += pointsInter[i].x;
    sumY[clusterId] += pointsInter[i].y;
    nPoints[clusterId] += 1;
  }

  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorCentroids = bufferCentroids.get_access<sycl_write>(cgh);
    auto accessornPoints = buffernPoints.get_access<sycl_read>(cgh);
    auto accessorSumX = bufferSumX.get_access<sycl_read>(cgh);
    auto accessorSumY = bufferSumY.get_access<sycl_read>(cgh);

    auto kern2 = [=](cl::sycl::id<1> id) {
      accessorCentroids[id].x = accessorSumX[id]/accessornPoints[id];
      accessorCentroids[id].y = accessorSumY[id]/accessornPoints[id];
    };
    cgh.parallel_for<class kernel2>(numOfK, kern2);
  });

}

int main(int argc, char** argv) { 
  std::string numOflen = argv[1];
  std::string numOfK = argv[2];

  size_t len = stoi(numOflen);
  size_t K = stoi(numOfK);
  size_t epoch = 100;

  std::cout << "Len: " << len << " " << "K: " << K << std::endl;

  Point* points;
  points = new Point[len];
   
  Point* centroids;
  centroids = new Point[K];

  int* nPoints = new int[K];
  float* sumX  = new float[K];
  float* sumY  = new float[K];
  
  int kk = 0;

  for (size_t i=0;i<len;i++) {
    Point p1 = Point(rand()%len, rand()%len);

    points[i] = p1;
    if (rand() % 100 > 50 && kk<K) {
      centroids[kk] = points[i];
      kk++;
    }
  }

  if (kk!=K) {
    for (size_t p=0;p<K;p++){
      centroids[p] = points[p];
    }
  }


  auto start = std::chrono::steady_clock::now();
  for (size_t iter = 0 ; iter < epoch ; iter++)
    kmean(points, centroids, len, K, nPoints, sumX, sumY);
  auto end = std::chrono::steady_clock::now();

  auto time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();


  std::cout << "Time: " << time << " milliseconds" << std::endl;

  return 0;
}
