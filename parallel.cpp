#include <omp.h>
#include <thread>
#include <vector>
#include <mutex>
#include <cstring>
#include <cstdlib>
#include <barrier>
#include <iostream>
#include <type_traits>

#define STEPS 100000000
#define SIZE 64u

using namespace std;

static unsigned num_treads = thread::hardware_concurrency();
unsigned get_num_threads()
{
    return num_treads;
}

void set_num_threads(unsigned t)
{
    num_treads = t;
    omp_set_num_threads(t);
}

struct partial_sum
{
    alignas(64) double Value;
};

typedef double (*function)(double);

typedef double (*unary_function)(double);

double Quadratic(double x)
{
    return x * x;
}

double IntegratePartialSum(function F, double a, double b)
{
    double Result = 0;
    double dx = (b - a) / STEPS;
    unsigned int T = get_num_threads();
    auto Vec = vector(T, partial_sum{ 0.0 });
    vector<thread> Threads;

    auto ThreadProcedure = [dx, T, F, a, &Vec](auto t)
    {
        for (auto i = t; i < STEPS; i += T)
            Vec[t].Value += F(dx * i + a);
    };

    for (unsigned t = 1; t < T; t++)
        Threads.emplace_back(ThreadProcedure, t);

    ThreadProcedure(0);
    for (auto& Thread : Threads)
        Thread.join();

    for (auto Elem : Vec)
        Result += Elem.Value;

    Result *= dx;
    return Result;
}

struct experiment_result
{
    double Result;
    double TimeInMs;
};


typedef double (*IntegrateFunction) (function, double, double);
experiment_result RunExperiment(IntegrateFunction I)
{
    double t0 = omp_get_wtime();
    double res = I(Quadratic, -1, 1);
    double t1 = omp_get_wtime();

    experiment_result Result;
    Result.Result = res;
    Result.TimeInMs = t1 - t0;

    return Result;
}

void ShowExperimentResult(IntegrateFunction I)
{
    set_num_threads(1);

    printf("%10s %10s %10s %14s\n", "Threads", "Result", "Time in ms", "Acceleration");
    experiment_result Experiment;
    Experiment = RunExperiment(I);
    printf("%10d %10g %10g %14g\n", 1, Experiment.Result, Experiment.TimeInMs, 1.0f);
    double Time = Experiment.TimeInMs;

    for (unsigned T = 2; T <= omp_get_num_procs(); T++)
    {
        set_num_threads(T);
        Experiment = RunExperiment(I);
        printf("%10d %10g %10g %14g\n", T, Experiment.Result, Experiment.TimeInMs, Time / Experiment.TimeInMs);
    }
    printf("\n");
}

double IntegrateFalseSharingOMP(function Function, double a, double b)
{
    unsigned int T;
    double Result = 0;
    double dx = (b - a) / STEPS;
    double* Accum = 0;

#pragma omp parallel shared(Accum, T)
    {
        unsigned int t = (unsigned int)omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned int)omp_get_num_threads();
            Accum = (double*)calloc(T, sizeof(double));
        }

        for (unsigned int i = t; i < STEPS; i += T)
            Accum[t] += Function(dx * i + a);
    }

    for (unsigned int i = 0; i < T; i++)
        Result += Accum[i];

    Result *= dx;
    free(Accum);
    return Result;
}

double IntegrateAlignOMP(function Function, double a, double b)
{
    unsigned int T;
    double Result = 0;
    double dx = (b - a) / STEPS;
    partial_sum* Accum = 0;

#pragma omp parallel shared(Accum, T)
    {
        unsigned int t = (unsigned int)omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned int)omp_get_num_threads();
            Accum = (partial_sum*)_aligned_malloc(T * sizeof(*Accum), SIZE);
            memset(Accum, 0, T * sizeof(*Accum));
        }

        for (unsigned int i = t; i < STEPS; i += T)
            Accum[t].Value += Function(dx * i + a);
    }

    for (unsigned int i = 0; i < T; i++)
        Result += Accum[i].Value;

    _aligned_free(Accum);
    return Result *= dx;
}

double IntegrateParallelOMP(function Function, double a, double b)
{
    double Result = 0;
    double dx = (b - a) / STEPS;

#pragma omp parallel
    {
        double Accum = 0;
        unsigned int t = (unsigned int)omp_get_thread_num();
        unsigned int T = (unsigned int)omp_get_num_threads();
        for (unsigned int i = t; i < STEPS; i += T)
            Accum += Function(dx * i + a);
#pragma omp critical
        Result += Accum;
    }

    Result *= dx;
    return Result;
}

double IntegrateReductionOMP(function Function, double a, double b)
{
    double Result = 0;
    double dx = (b - a) / STEPS;

#pragma omp parallel for reduction(+:Result)
    for (int i = 0; i < STEPS; i++)
        Result += Function(dx * i + a);

    Result *= dx;
    return Result;
}

double IntegrateMutex(function function, double a, double b) {
    unsigned T = get_num_threads();
    mutex mtx;
    vector<thread> threads;
    double result = 0, dx = (b - a) / STEPS;

    for (unsigned t = 0; t < T; t++) {
        threads.emplace_back([=, &result, &mtx]() {
            double R = 0;
            for (unsigned i = t; i < STEPS; i += T) {
                R += function(i * dx + a);
            }
            scoped_lock lock{ mtx };
            result += R;
            });

    }
    for (auto& thr : threads) {
        thr.join();
    }

    return result * dx;
}

double IntegrateAtomic(function Function, double a, double b) {
    vector<thread> threads;
    atomic<double> result = { 0 };
    double dx = (b - a) / STEPS;
    unsigned int T = get_num_threads();

    auto fun = [dx, &result, Function, a, T](auto t) {
        double R = 0;
        for (unsigned i = t; i < STEPS; i += T) {
            R += Function(i * dx + a);
        }

        result += R;
    };

    for (unsigned int t = 1; t < T; ++t) {
        threads.emplace_back(fun, t);
    }

    fun(0);

    for (auto& thr : threads) {
        thr.join();
    }

    return result * dx;
}

auto ceil_div(auto x, auto y)
{
    return (x + y - 1) / y;
}

template <class ElementType, class BinaryFn>
ElementType reduce_vector(const ElementType* V, size_t n, BinaryFn f, ElementType zero)
{
    unsigned T = get_num_threads();

    struct reduction_partial_result_t
    {
        alignas(hardware_destructive_interference_size) ElementType value;
    };
    static auto reduction_partial_results =
        vector<reduction_partial_result_t>(T, reduction_partial_result_t{ zero });
    constexpr size_t k = 2;
    barrier<> bar{ T };

    auto thread_proc = [=, &bar](unsigned t)
    {
        auto K = ceil_div(n, k);
        size_t Mt = K / T, it1 = K % T;
        if (t < it1)
            it1 = ++Mt * t;
        else
            it1 += Mt * t;
        it1 *= k;
        size_t mt = Mt * k;
        auto it2 = it1 + mt;
        ElementType accum = zero;
        for (size_t i = it1; i < it2; ++i)
            accum = f(accum, V[i]);
        reduction_partial_results[t].value = accum;
        for (size_t s = 1u, s_next = 2u; s < T; s = s_next, s_next += s_next)
        {
            bar.arrive_and_wait();
            if ((t % s_next) == 0 && s + t < T)
                reduction_partial_results[t].value = f(reduction_partial_results[t].value, reduction_partial_results[t + s].value);
        }
    };

    vector<thread> threads;
    for (unsigned t = 1; t < T; ++t)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for (auto& thread : threads)
        thread.join();

    return reduction_partial_results[0].value;
}

template <class ElementType, class UnaryFn, class BinaryFn>
requires (
    is_invocable_r_v<ElementType, UnaryFn, ElementType>&&
    is_invocable_r_v<ElementType, BinaryFn, ElementType, ElementType>
    )
    ElementType reduce_range(ElementType a, ElementType b, size_t n, UnaryFn get, BinaryFn reduce_2, ElementType zero)
{
    unsigned T = get_num_threads();
    struct reduction_partial_result_t
    {
        alignas(hardware_destructive_interference_size) ElementType value;
    };
    static auto reduction_partial_results =
        vector<reduction_partial_result_t>(thread::hardware_concurrency(), reduction_partial_result_t{ zero });

    barrier<> bar{ T };
    constexpr size_t k = 2;
    auto thread_proc = [=, &bar](unsigned t)
    {
        auto K = ceil_div(n, k);
        double dx = (b - a) / n;
        size_t Mt = K / T;
        size_t it1 = K % T;

        if (t < it1)
        {
            it1 = ++Mt * t;
        }
        else
        {
            it1 = Mt * t + it1;
        }
        it1 *= k;
        size_t mt = Mt * k;
        size_t it2 = it1 + mt;

        ElementType accum = zero;
        for (size_t i = it1; i < it2; i++)
            accum = reduce_2(accum, get(a + i * dx));

        reduction_partial_results[t].value = accum;

        for (size_t s = 1, s_next = 2; s < T; s = s_next, s_next += s_next)
        {
            bar.arrive_and_wait();
            if (((t % s_next) == 0) && (t + s < T))
                reduction_partial_results[t].value = reduce_2(reduction_partial_results[t].value,
                    reduction_partial_results[t + s].value);
        }
    };

    vector<thread> threads;
    for (unsigned t = 1; t < T; t++)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for (auto& thread : threads)
        thread.join();
    return reduction_partial_results[0].value;
}

double IntegrateReductionReduceRange(unary_function F, double a, double b)
{
    double dx = (b - a) / STEPS;
    return reduce_range(a, b, STEPS, F, [](auto x, auto y) {return x + y; }, 0.0) * dx;
}



double IntegrateReductionStatic(function Function, double a, double b)
{
    double Result = 0;
    double dx = (b - a) / STEPS;

#pragma omp parallel for reduction(+:Result) schedule(static)
    for (int i = 0; i < STEPS; i++)
        Result += Function(dx * i + a);

    Result *= dx;
    return Result;
}

double IntegrateReductionDynamic(function Function, double a, double b)
{
    double Result = 0;
    double dx = (b - a) / STEPS;

#pragma omp parallel for reduction(+:Result) schedule(dynamic)
    for (int i = 0; i < STEPS; i++)
        Result += Function(dx * i + a);

    Result *= dx;
    return Result;
}

double RandomizeArraySingleThread(uint64_t seed, uint32_t* arr, double arrLength, uint32_t min, uint32_t max)
{
    uint64_t a = 6364136223846793005;
    uint64_t b = 1;
    uint64_t prevNumber = seed;
    double result = 0;

    for (uint32_t i = 0; i < arrLength; i++)
    {
        uint64_t nextNumber = a * prevNumber + b;
        arr[i] = ((nextNumber % (max - min + 1)) + min);
        prevNumber = nextNumber;
        result += arr[i];
    }
    return result / arrLength;
}

uint64_t pow64(uint64_t a, unsigned b)
{
    uint64_t result = a;
    if (b == 0) return 1;
    for (unsigned i = 1; i < b; i++)
    {
        result *= a;
    }
    return result;
}

uint64_t sumA(uint64_t a, unsigned T)
{
    uint64_t result = 0;
    for (unsigned i = 0; i < T; i++)
    {
        result += pow64(a, i);
    }
    return result;
}

double RandomizeArrayThreads(uint64_t seed, uint32_t* arr, double arrLength, uint32_t min, uint32_t max)
{
    unsigned T = get_num_threads();
    uint64_t a = 6364136223846793005;
    uint64_t aT = pow64(a, T);
    uint64_t b = 1;
    uint64_t resSumAB = b * sumA(a, T);
    uint64_t prevN = seed;
    uint64_t* result = new uint64_t[T];

    uint64_t* tmp_arr = new uint64_t[T];

    for (unsigned i = 0; i < T; i++)
    {
        result[i] = 0;
    }

    for (unsigned i = 0; i < T; i++)
    {
        uint64_t nextNumber = a * prevN + b;
        arr[i] = ((nextNumber % (max - min + 1)) + min);
        prevN = nextNumber;
        tmp_arr[i] = nextNumber;
        result[i] += arr[i];
    }
    vector<thread> threads;
    uint64_t R = 0;

    if (T > arrLength)
    {
        for (unsigned t = 0; t < arrLength; t++)
        {
            R += result[t];
        }

        return (double)R / (double)arrLength;
    }

    auto createElems = [&arr, seed, a, b, T, resSumAB, min, max, aT, arrLength, &result, tmp_arr](auto t)
    {
        uint64_t prevNumber = tmp_arr[t];
        uint64_t tmp_result = 0;
        for (auto i = t + T; i < arrLength; i += T)
        {
            uint64_t num = aT * prevNumber + resSumAB;
            arr[i] = ((num % (max - min + 1)) + min);
            prevNumber = num;
            tmp_result += arr[i];
        }

        result[t] += tmp_result;

    };

    for (unsigned t = 0; t < T; t++)
        threads.emplace_back(createElems, t);

    for (auto& thread : threads)
        thread.join();

    for (unsigned t = 0; t < T; t++)
    {
        R += result[t];
    }

    return (double)R / (double)arrLength;

}

typedef double (*randomize_function)(uint64_t, uint32_t*, double, uint32_t, uint32_t);

struct randomization_experiment_result
{
    double Average;
    double ExpectedAverage;
    double DifferenceAverage;
    double Time;
};

randomization_experiment_result RunRandomizationExperiment(randomize_function Randomize, uint32_t* Array, uint32_t ArrayLength)
{
    uint64_t Seed = 93821;
    uint32_t Min = 100;
    uint32_t Max = 1500;

    double ExpectedAverage = 0.5f * (double)(Min + Max);

    double T0 = omp_get_wtime();
    double Average = Randomize(Seed, Array, ArrayLength, Min, Max);
    double T1 = omp_get_wtime();

    randomization_experiment_result Result;
    Result.Average = Average;
    Result.ExpectedAverage = ExpectedAverage;
    Result.DifferenceAverage = Result.ExpectedAverage - Result.Average;
    Result.Time = T1 - T0;

    return Result;
}


void ShowRandomizationExperimentResult(randomize_function Randomize, uint32_t* Array, uint32_t ArrayLength, const char* Name)
{
    printf("%s (%d elements)\n", Name, ArrayLength);
    set_num_threads(1);

    uint32_t Width = 4;
    printf("%-*s %-*s %-*s %-*s %-*s %-*s\n",
        (int)(strlen("Threads") + Width), "Threads",
        (int)(strlen("Expected Average") + Width), "Expected Average",
        (int)(strlen("Average") + Width), "Average",
        (int)(strlen("DifferenceAverage") + Width), "DifferenceAverage",
        (int)(strlen("Time(s)") + Width), "Time(s)",
        (int)(strlen("Acceleration") + Width), "Acceleration");

    randomization_experiment_result Experiment;
    Experiment = RunRandomizationExperiment(Randomize, Array, ArrayLength);
    printf("%-*u %-*f %-*f %-*f %-*f %-*f\n",
        (int)(strlen("Threads") + Width), 1,
        (int)(strlen("Expected Average") + Width), Experiment.ExpectedAverage,
        (int)(strlen("Average") + Width), Experiment.Average,
        (int)(strlen("DifferenceAverage") + Width), Experiment.DifferenceAverage,
        (int)(strlen("Time(s)") + Width), Experiment.Time,
        (int)(strlen("Acceleration") + Width), 1.0);

    double SingleThreadedTime = Experiment.Time;
    for (unsigned T = 2; T <= omp_get_num_procs(); T++)
    {
        set_num_threads(T);
        Experiment = RunRandomizationExperiment(Randomize, Array, ArrayLength);
        printf("%-*u %-*f %-*f %-*f %-*f %-*f\n",
            (int)(strlen("Threads") + Width), T,
            (int)(strlen("Expected Average") + Width), Experiment.ExpectedAverage,
            (int)(strlen("Average") + Width), Experiment.Average,
            (int)(strlen("DifferenceAverage") + Width), Experiment.DifferenceAverage,
            (int)(strlen("Time(s)") + Width), Experiment.Time,
            (int)(strlen("Acceleration") + Width), SingleThreadedTime / Experiment.Time);
    }
    printf("\n");
}

void main()
{

    uint32_t* Array = new uint32_t[200000000];
    double ArrayLength = 200000000;
    ShowRandomizationExperimentResult(RandomizeArraySingleThread, Array, ArrayLength, "one");
    uint32_t* Array2 = new uint32_t[200000000];
    ShowRandomizationExperimentResult(RandomizeArrayThreads, Array2, ArrayLength, "many");

    printf("IntegrateAtomic\n");
    ShowExperimentResult(IntegrateAtomic);

    printf("IntegrateMutex\n");
    ShowExperimentResult(IntegrateMutex);

    printf("IntegratePartialSum\n");
    ShowExperimentResult(IntegratePartialSum);

    printf("IntegrateAlignOMP\n");
    ShowExperimentResult(IntegrateAlignOMP);

    printf("IntegrateParallelOMP\n");
    ShowExperimentResult(IntegrateParallelOMP);

    printf("IntegrateFalseSharingOMP\n");
    ShowExperimentResult(IntegrateFalseSharingOMP);

    printf("IntegrateReductionReduceRange\n");
    ShowExperimentResult(IntegrateReductionReduceRange);

    printf("IntegrateReductionOMP\n");
    ShowExperimentResult(IntegrateReductionOMP);

    printf("IntegrateReductionDynamic\n");
    ShowExperimentResult(IntegrateReductionDynamic);

    printf("IntegrateReductionStatic\n");
    ShowExperimentResult(IntegrateReductionStatic);

}