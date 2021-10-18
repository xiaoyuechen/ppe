#include <sys/time.h>
#include <random.h>

#define ITERS 1000000000
#define NUMBER_OF_RUNS 10

struct timeval start_time, end_time;

int *arraySizes = [64, 128, 1024, 65536, 33554432];

void cpu_8Add() {

    char sum = 1;

    __m128i sum_v0 = _mm_set1_epi8(0);
    __m128i sum_v1 = _mm_set1_epi8(0);
    __m128i sum_v2 = _mm_set1_epi8(0);
    __m128i sum_v3 = _mm_set1_epi8(0);
    __m128i addthis_v = _mm_set1_epi8(1);

    for (int run=0; run<NUMBER_OF_RUNS; run++) {

        gettimeofday(&start_time, NULL);

        for (long test=0; test<ITERS; test+=16*4) {
            sum_v0 = _mm256_add_epi8(addthis_v, sum_v0);
            sum_v1 = _mm256_add_epi8(addthis_v, sum_v1);
            sum_v2 = _mm256_add_epi8(addthis_v, sum_v2);
            sum_v3 = _mm256_add_epi8(addthis_v, sum_v3);
        }

        gettimeofday(&end_time, NULL);

        double time_in_sec = (end_time.tv_sec+end_time.tv_usec/1000000.0) - (start_time.tv_sec+start_time.tv_usec/1000000.0);
        double GOPS = (ITERS/time_in_sec)/1000000000;
        for (int i=0; i<16;i++)
            sum+=((char*)(&sum_v0))[i]+((char*)(&sum_v1))[i]+((char*)(&sum_v2))[i]+((char*)(&sum_v3))[i];
        printf("sum: %d\n", sum);
        printf("Completed %lu adds in %g seconds for %g GOPS.\n", ITERS, time_in_sec, GOPS);
    }
}

void cpu_FPAdd() {

    float sum = 1.0;

    __m128i sum_v0 = _mm_set1_ps(0.0);
    __m128i sum_v1 = _mm_set1_ps(0.0);
    __m128i sum_v2 = _mm_set1_ps(0.0);
    __m128i sum_v3 = _mm_set1_ps(0.0);
    __m128i addthis_v = _mm_set1_ps(1.0);

    for (int run=0; run<NUMBER_OF_RUNS; run++) { 

        gettimeofday(&start_time, NULL);

        for (long test=0; test<ITERS; test+=16*4) { 
            sum_v0 = _mm256_add_ps(addthis_v, sum_v0);
            sum_v1 = _mm256_add_ps(addthis_v, sum_v1);
            sum_v2 = _mm256_add_ps(addthis_v, sum_v2);
            sum_v3 = _mm256_add_ps(addthis_v, sum_v3);
        } 

        gettimeofday(&end_time, NULL);

        double time_in_sec = (end_time.tv_sec+end_time.tv_usec/1000000.0) - (start_time.tv_sec+start_time.tv_usec/1000000.0);
        double GOPS = (ITERS/time_in_sec)/1000000000;
        for (int i=0; i<16;i++)
            sum+=((float*)(&sum_v0))[i]+((float*)(&sum_v1))[i]+((float*)(&sum_v2))[i]+((float*)(&sum_v3))[i];
        printf("sum: %d\n", sum);
        printf("Completed %lu adds in %g seconds for %g GOPS.\n", ITERS, time_in_sec, GOPS);
    } 
}

void run_cpu_seqLoad() {
    for (int i=0; i<arraySizes; i++) {
        int arraySize = arraySizes[i];
        int *array[arraySize] = {0};
        
        cpu_seqLoad(array,arraySize);
    }
}


void cpu_seqLoad(int *array, int numElems) {
   
    int elem;
   
    for (int run=0; run<NUMBER_OF_RUNS; run++) {

        gettimeofday(&start_time, NULL);
 
        for (long test=0; test<ITERS; test++) {
      		  elem = array[test % numElems];
        }

        gettimeofday(&end_time, NULL);

        double time_in_sec = (end_time.tv_sec+end_time.tv_usec/1000000.0) - (start_time.tv_sec+start_time.tv_usec/1000000.0);
        double GOPS = (ITERS/time_in_sec)/1000000000;
        printf("Completed %lu loads in %g seconds for %g GOPS.\n", ITERS, time_in_sec, GOPS);

    }  
    
}

void run_cpu_randLoad() {
    int tmp;
    int r;

    for (int i=0; i<arraySizes; i++) {
        int arraySize = arraySizes[i];
        int *array[arraySize];
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 gen(rd()); // seed the generator
        std::uniform_int_distribution<> distr(0, arraySize); // define the range

        // set each element to its index
        for (int k=0; i<arraySize; k++) {
        		array[k] = k;
        }
        
        // for each element, swap with a random element
        for (int j=0; j<arraySize; j++) {
            r = distr(gen);
            tmp = array[r];
            array[r] = array[j];
            array[j] = r;
        }

        cpu_randLoad(array);
    }
}

void cpu_randLoad(int *array) {
 
    int elem = 0;

    for (int run=0; run<NUMBER_OF_RUNS; run++) {

        gettimeofday(&start_time, NULL);

        for (long test=0; test<ITERS; test++) {
            elem = array[elem];
        }

        gettimeofday(&end_time, NULL);

        double time_in_sec = (end_time.tv_sec+end_time.tv_usec/1000000.0) - (start_time.tv_sec+start_time.tv_usec/1000000.0);
        double GOPS = (ITERS/time_in_sec)/1000000000;
        printf("Completed %lu loads in %g seconds for %g GOPS.\n", ITERS, time_in_sec, GOPS);

    }

} 
   

