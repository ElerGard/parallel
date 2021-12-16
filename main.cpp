#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <omp.h>
#include <atomic>
#include <cstdlib>
#include <fstream>

#define STEPS 100000000
#define CACHE_LINE 64u
#define A -1
#define B 1

unsigned get_num_threads();

void set_num_threads(unsigned T);

typedef double (*f_t)(double);

typedef double (*E_t)(double, double, f_t);

struct ExperimentResult {
    double result;
    double time;
};
struct partialSumT {
    alignas(64) double val;
};

typedef double (*I_t)(double, double, f_t);

using namespace std;

double f(double x) {
    return x * x;
}

double integrateDefault(double a, double b, f_t f) {
    double result = 0, dx = (b - a) / STEPS;

    for (unsigned i = 0; i < STEPS; i++) {
        result += f(i * dx + a);
    }

    return result * dx;
}

double integrateCrit(double a, double b, f_t f) {
    double result = 0, dx = (b - a) / STEPS;

#pragma omp parallel
    {
        double R = 0;
        unsigned t = (unsigned) omp_get_thread_num();
        unsigned T = (unsigned) omp_get_num_threads();

        for (unsigned i = t; i < STEPS; i += T) {
            R += f(i * dx + a);
        }
#pragma omp critical
        result += R;
    }
    return result * dx;
}

double integrateMutex(double a, double b, f_t f) {
    unsigned T = thread::hardware_concurrency();
    mutex mtx;
    vector<thread> threads;
    double result = 0, dx = (b - a) / STEPS;

    for (unsigned t = 0; t < T; t++) {
        threads.emplace_back([=, &result, &mtx]() {
            double R = 0;
            for (unsigned i = t; i < STEPS; i += T) {
                R += f(i * dx + a);
            }

            {
                scoped_lock lck{mtx};
                result += R;
            }
        });

    }
    for (auto &thr: threads) {
        thr.join();
    }

    return result * dx;
}

double integrateArr(double a, double b, f_t f) {
    unsigned T;
    double result = 0, dx = (b - a) / STEPS;
    double *accum;

#pragma omp parallel shared(accum, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            accum = (double *) calloc(T, sizeof(double));
            //accum1.reserve(T);
        }

        for (unsigned i = t; i < STEPS; i += T) {
            accum[t] += f(dx * i + a);
        }
    }

    for (unsigned i = 0; i < T; ++i) {
        result += accum[i];
    }

    free(accum);

    return result * dx;
}

double integrateArrAlign(double a, double b, f_t f) {
    unsigned T;
    double result = 0, dx = (b - a) / STEPS;
    partialSumT *accum;

#pragma omp parallel shared(accum, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            accum = (partialSumT *) aligned_alloc(CACHE_LINE, T * sizeof(partialSumT));
        }

        for (unsigned i = t; i < STEPS; i += T) {
            accum[t].val += f(dx * i + a);
        }
    }

    for (unsigned i = 0; i < T; ++i) {
        result += accum[i].val;
    }

    free(accum);

    return result * dx;
}

double integrateReduction(double a, double b, f_t f) {
    double result = 0, dx = (b - a) / STEPS;

#pragma omp parallel for reduction(+: result)
    for (unsigned int i = 0; i < STEPS; ++i) {
        result += f(dx * i + a);
    }

    return result * dx;
}

double integratePS(double a, double b, f_t f) {
    double dx = (b - a) / STEPS;
    double result = 0;
    unsigned T = omp_get_num_threads();
    auto vec = vector(T, partialSumT{0.0});
    vector<thread> threadVec;

    auto threadProc = [=, &vec](auto t) {
        for (auto i = t; i < STEPS; i += T) {
            vec[t].val += f(dx * i + a);
        }
    };

    for (auto t = 1; t < T; t++) {
        threadVec.emplace_back(threadProc, t);
    }

    threadProc(0);

    for (auto &thread: threadVec) {
        thread.join();
    }

    for (auto elem: vec) {
        result += elem.val;
    }

    return result * dx;
}

double integrateAtomic(double a, double b, f_t f) {
    vector<thread> threads;
    std::atomic<double> result = {0};
    double dx = (b - a) / STEPS;
    unsigned int T = omp_get_num_threads();

    auto fun = [dx, &result, f, a, T](auto t) {
        double R = 0;
        for (unsigned i = t; i < STEPS; i += T) {
            R += f(i * dx + a);
        }

        result = result + R;
    };

    for (unsigned int t = 1; t < T; ++t) {
        threads.emplace_back(fun, t);
    }

    fun(0);

    for (auto &thr: threads) {
        thr.join();
    }

    return result * dx;
}

ExperimentResult runExperiment(I_t I) {
    double t0, t1, result;

    t0 = omp_get_wtime();
    result = I(A, B, f);
    t1 = omp_get_wtime();

    return {result, t1 - t0};
}

void showExperimentResults(I_t I, ofstream &file) {
    omp_set_num_threads(1);
    ExperimentResult R = runExperiment(I);
    double T1 = R.time;

    file.width(10);
    file << "Threads,";
    file.width(10);
    file << "Time,";
    file.width(14);
    file << "Acceleration,";
    file.width(10);
    file << "Result" << endl;
    printf("%10s,%10s,%14s,%10s\n", "Threads", "Time", "Acceleration", "Result");

    file.width(10);
    file << "1,";
    file.width(10);
    file.precision(6);
    file << to_string(R.time) + ",";
    file.width(14);
    file.precision(6);
    file << to_string(T1 / R.time) + ",";
    file.width(10);
    file.precision(6);
    file << to_string(R.result) << endl;
    printf("%10d,%10g,%14g,%10g\n", 1, R.time, T1 / R.time, R.result);

    for (int T = 2; T <= omp_get_num_procs(); ++T) {
        omp_set_num_threads(T);
        ExperimentResult result = runExperiment(I);
        file.width(9);
        file << T << ",";
        file.width(10);
        file.precision(6);
        file << to_string(result.time) + ",";
        file.width(14);
        file.precision(6);
        file << to_string(T1 / result.time) + ",";
        file.width(10);
        file.precision(6);
        file << to_string(result.result) << endl;
        // printf("%10g\t%10g\t%10g\n", result.result, result.time, T1/result.time);
        printf("%10d,%10g,%14g,%10g\n", T, result.time, T1 / result.time, result.result);
    }

    file << endl;
    cout << endl;
}

int main() {
    ifstream _file;
    ofstream file;
    int i = 1;
    for (i; i < 20; i++) {
        string str = "results" + to_string(i) + ".txt";
        _file.open(str);
        if (!_file.good()) {
            _file.close();
            cout << "file create\n";
            file.open(str);
            i = 20;
        } else
            _file.close();
    }
    file << "integrateDefault" << endl;
    showExperimentResults(integrateDefault, file);
    file << "integrateCrit" << endl;
    showExperimentResults(integrateCrit, file);
    file << "integrateMutex" << endl;
    showExperimentResults(integrateMutex, file);
    file << "integrateArr" << endl;
    showExperimentResults(integrateArr, file);
    file << "integrateArrAlign" << endl;
    showExperimentResults(integrateArrAlign, file);
    file << "integrateReduction" << endl;
    showExperimentResults(integrateReduction, file);
    file << "integratePS" << endl;
    showExperimentResults(integratePS, file);
    file << "integrateAtomic" << endl;
    showExperimentResults(integrateAtomic, file);

    return 0;
}
