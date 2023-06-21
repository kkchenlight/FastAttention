/*
 * @Description:
 * @Author: kkchen
 * @Email: kkchen.lg@qq.com
 * @Date: 2023-06-10 08:07:11
 * @LastEditTime: 2023-06-21 20:18:31
 * @LastEditors: kkchen
 */
#include <float.h>
#include <math.h>

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using namespace std;
using namespace chrono;

class Clock {
 public:
  Clock(string& exp_name) {
    start = chrono::steady_clock::now();
    expname = exp_name;
  };
  ~Clock() {
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    chrono::microseconds duration =
        chrono::duration_cast<microseconds>(end - start);
    printf("%s cost %f ms\n", expname.c_str(), duration.count() / 1000.f);
  }

 public:
  chrono::steady_clock::time_point start;
  std::string expname;
};

void navieSoftmax(float* dst, float* src, int data_len) {
  // 1. get max
  float max_value = -FLT_MIN;  // set it to MIN_FLOAT
  for (int i = 0; i < data_len; i++) {
    if (src[i] > max_value) {
      max_value = src[i];
    }
  }

  // 2. get sum
  float sum = 0.f;
  for (int i = 0; i < data_len; i++) {
    sum += std::expf(src[i] - max_value);
  }

  // 3. caculate output
  for (int i = 0; i < data_len; i++) {
    dst[i] = std::expf(src[i] - max_value) / sum;
  }
  // printf("navie softmax Done!\n");
  // printf("max = %f, sum = %f\n", max_value, sum);

  return;
}

void fastSoftmax(float* dst, float* src, int data_len) {
  float old_max = -FLT_MAX;
  float sum = 0.0f;
  for (int i = 0; i < data_len; i++) {
    float new_max = std::max(old_max, src[i]);
    sum = sum * std::expf(old_max - new_max) + std::exp(src[i] - new_max);
    old_max = new_max;
  }
  for (int i = 0; i < data_len; i++) {
    dst[i] = std::expf(src[i] - old_max) / sum;
  }
  // printf("max = %f, sum = %f\n", old_max, sum);
  //  printf("fastSoftmax Done!\n");
}

void checkResult(float* src, int data_len) {
  float sum = 0.f;
  for (int i = 0; i < data_len; i++) {
    sum += src[i];
  }

  if (std::abs(sum - 1.0) < 0.0001f) {
    printf("pass!!!!\n");
  } else {
    printf("fail, sum = %f\n", sum);
  }
}

int main(int argc, char** argv) {
  int data_len = atoi(argv[1]);
  int times = atoi(argv[2]);
  random_device de;
  std::mt19937 gen(de());
  std::uniform_real_distribution<float> dis(0, 1.0);
  printf("data_len: %d, times: %d\n", data_len, times);
  // vector<float> data_vec;
  float* src = new float[data_len];
  for (int i = 0; i < data_len; i++) {
    src[i] = dis(gen);
  }

  float* navie_ptr = new float[data_len];
  float* fast_ptr = new float[data_len];

  {
    string expname = "navie";
    Clock a = Clock(expname);
    for (int i = 0; i < times; i++) {
      navieSoftmax(navie_ptr, src, data_len);
    }
  }
  checkResult(navie_ptr, data_len);

  {
    string expname = "fast";
    Clock a = Clock(expname);
    for (int i = 0; i < times; i++) {
      fastSoftmax(fast_ptr, src, data_len);
    }
  }
  checkResult(fast_ptr, data_len);

  delete[] navie_ptr;
  delete[] fast_ptr;
  delete[] src;
}
