#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

const int nproc = 9;
// количество процессоров по оси X.
const int nxproc = 3;
// количество процессоров по оси Y.
const int nyproc = 3;
 
// количество строк. 
const int N = 10;
// количество столбцов.
const int M = 10;

// область определения функции.
const int A1 = -1;
const int A2 = 2;
const int B1 = -2;
const int B2 = 2;

const float EPS = 1e-;
const float h1 = float((A2 - A1)) / float(M);
const float h2 = float((B2 - B1)) / float(N);

static float ro[N+1][M+1];
static float diff[N+1][M+1];
static float function_value[N+1][M+1];
static float w[N+1][M+1] = {0.0};
static float Aw[N+1][M+1] = {0.0};
static float Ar[N+1][M+1] = {0.0};
static float w_new[N+1][M+1] = {0.0};
static float residuals[N+1][M+1] = {0.0}; 


float estimation_of_accuracy(float w_true[N+1][M+1], float w_est[N+1][M+1]) {
	float mean = 0.0;
	float max = 0.0;
	for (int i=0; i<N+1; i++) {
		for (int j=0; j<M+1; j++) {
			float val = fabs(w_true[i][j] - w_est[i][j]);
			mean += val;
			if (val > max)
				max = val;
		}
	}
	mean /= ((N+1)*(M+1));
	return max; 
};

// печать массива так, будто это координатная плоскость.
void print_array(float arr[][N+1]) {
	for(int i=0; i<N+1; i++) {
		for(int j=0; j<M+1; j++) {
			printf("%f ", arr[N-i][j]);
		}
		printf("\n");
	}
	printf("\n\n");
};	

void boundary_init(float arr[N+1][M+1]){

	for (int i=0; i<N+1; i++) {
		for (int j=0; j<M+1; j++) {
			float x = A1 + j*h1;
			float y = B1 + i*h2;
			// bottom
			if (i == 0)
				arr[i][j] = exp(1 - pow(x + B1, 2));	
			// top
			else if (i == N)
				arr[i][j] = exp(1 - pow(x + B2, 2));	
			// left
			else if (j == 0)
				arr[i][j] = exp(1 - pow(y + A1, 2));
			// right
			else if (j == M)
				arr[i][j] = exp(1 - pow(y + A2, 2));
		}
	}		
}


int main(int argc, char **argv) {

	bool error = true;
	float tau;
	float diff_norm;
	int num_procs, my_id;
	float B[N+1][M+1] = {0.0};
	time_t t0 = time(0);

	// получаем значения искомой функции на сетке.
	for (int i=0; i<N+1; i++) {
		for (int j=0; j<M+1; j++) {
			float x = A1 + j*h1;
			float y = B1 + i*h2;
			function_value[i][j] = exp(1 - pow(x+y, 2));
		}
	}		
	
	// создание матрицы "РО".
	for (int i=0; i<N+1; i++) {
		for (int j=0; j<M+1; j++) {
			ro[i][j] = 1.0;
			if (i == 0 || i == N)
				ro[i][j] *= 0.5;
			if (j == 0 || j == M)
				ro[i][j] *= 0.5;
		}
	};	

	// инициализируем граничные значения для нужных матриц.
	boundary_init(w);
	boundary_init(B);

	// создание матрицы B (правой части СЛАУ).
	for (int i=1; i<N; i++) {
		for (int j=1; j<M; j++) {
			float x = A1 + j*h1;
			float y = B1 + i*h2;	
			B[i][j] = expf(1 - powf(x+y, 2))*(16 + 6*x + 2*y - 8*powf(x+y,2)*(4+x));
			if (x + y > 0.0)
				B[i][j] += expf(1 - powf(x+y, 2))*powf(x+y, 2);		
		}
	}

	//print_array(B);

	MPI_Request req[4];

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   	int left_id = my_id - 1;
   	int right_id = my_id + 1;
   	int bot_id = my_id - nxproc;
   	int top_id = my_id + nxproc;

   	if (my_id % nxproc == 0)
   		left_id = MPI_PROC_NULL;
   	if (my_id % nxproc == nxproc - 1)
   		right_id = MPI_PROC_NULL;
   	if (my_id / nxproc == 0)
   		bot_id = MPI_PROC_NULL;
   	if (my_id / nxproc == nyproc - 1)
   		top_id = MPI_PROC_NULL;

   	int x_begin = 1+(my_id % nxproc)*((M-1) / nxproc);
   	int x_end;
	if (my_id % nxproc == nxproc - 1)
		x_end = M;
	else
		x_end = x_begin + (M-1)/nxproc;

	int y_begin = 1+(my_id / nxproc)*((N-1) / nyproc);
	int y_end;
	if (my_id / nxproc == nyproc - 1)
		y_end = N;
	else
		y_end = y_begin + (N-1)/nyproc;

	//printf("Hello world! I'm process %i and my border is:\n    (x: from %i to %i, y: from %i to %i)\n", my_id, x_begin, x_end, y_begin, y_end);

   	while (error) {

   		//printf("Hello world! I'm process %i\n", my_id);

   		// ПАРАЛЛЕЛЬНОЕ ВОЗДЕЙСТВИЕ ОПЕРАТОРА.

   		// y-axis
		MPI_Irecv(&w[y_end][x_begin], x_end - x_begin, MPI_FLOAT, top_id, 2001, MPI_COMM_WORLD, &req[0]);
		MPI_Irecv(&w[y_begin-1][x_begin], x_end - x_begin, MPI_FLOAT, bot_id, 2002, MPI_COMM_WORLD, &req[1]);
		MPI_Isend(&w[y_end-1][x_begin], x_end - x_begin, MPI_FLOAT, top_id, 2002, MPI_COMM_WORLD, &req[2]);	
		MPI_Isend(&w[y_begin][x_begin], x_end - x_begin, MPI_FLOAT, bot_id, 2001, MPI_COMM_WORLD, &req[3]);	
		MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

		//printf("Hello world! I'm process %i\n", my_id);
		
		// x-axis
		float *sbufleft, *sbufright, *rbufleft, *rbufright;
		int bufsize, posleft=0, posright=0;

		MPI_Pack_size(y_end - y_begin, MPI_FLOAT, MPI_COMM_WORLD, &bufsize);
		sbufleft = (float*) malloc(bufsize);
		sbufright = (float*) malloc(bufsize);
		rbufleft = (float*) malloc(bufsize);
		rbufright = (float*) malloc(bufsize);

		/* Pack the data before sending */
		for (int i=y_begin; i<y_end; i++) {
			MPI_Pack(&w[i][x_end-1], 1, MPI_FLOAT, sbufright, bufsize, &posright, MPI_COMM_WORLD);
			MPI_Pack(&w[i][x_begin], 1, MPI_FLOAT, sbufleft, bufsize, &posleft, MPI_COMM_WORLD);
		}

		//MPI_Barrier(MPI_COMM_WORLD);

		/* Execute now the real communication */
		MPI_Irecv(rbufleft, bufsize, MPI_PACKED, left_id, 2001, MPI_COMM_WORLD, &req[0]);
		MPI_Irecv(rbufright, bufsize, MPI_PACKED, right_id, 2002, MPI_COMM_WORLD, &req[1]);
		MPI_Isend(sbufleft, posleft, MPI_PACKED, left_id, 2002, MPI_COMM_WORLD, &req[2]);
		MPI_Isend(sbufright, posright, MPI_PACKED, right_id, 2001, MPI_COMM_WORLD, &req[3]);
		MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

		/* Unpack the received data */
		posright = posleft = 0;
		for (int i=y_begin; i<y_end; i++) {
			if (right_id != MPI_PROC_NULL)
				MPI_Unpack(rbufright, bufsize, &posright, &w[i][x_end], 1, MPI_FLOAT, MPI_COMM_WORLD);
			if (left_id != MPI_PROC_NULL)
				MPI_Unpack(rbufleft, bufsize, &posleft, &w[i][x_begin-1], 1, MPI_FLOAT, MPI_COMM_WORLD);
		}	

		//printf("Hello world! I'm process %i\n", my_id);

   		#pragma omp parallel for 
   		for (int i=y_begin; i<y_end; i++) {
			#pragma omp parallel for 
			for (int j=x_begin; j<x_end; j++) {
				float x = A1 + j*h1;
				float y = B1 + i*h2;

				Aw[i][j] = (1.0 / h1 * ((4.0 + x + 0.5*h1)*(w[i][j+1] - w[i][j]) / h1 - \
										(4.0 + x - 0.5*h1)*(w[i][j] - w[i][j-1]) / h1) + \
							1.0 / h2 * ((4.0 + x)*(w[i+1][j] - 2.0*w[i][j] + w[i-1][j]) / h2)) * (-1);
				if (x + y > 0.0)
					Aw[i][j] += powf(x + y, 2)*w[i][j];			  				  
			}
		}

		free(rbufleft);
		free(sbufleft);
		free(rbufright);
		free(sbufright);

   		// ПАРАЛЛЕЛЬНОЕ ВЫЧИСЛЕНИЕ НЕВЯЗКИ.

   		//if (my_id == 0)
   		//	print_array(residuals);

		for (int i=y_begin; i<y_end; i++) {
			for (int j=x_begin; j<x_end; j++) {
				residuals[i][j] = Aw[i][j] - B[i][j];
			}
		}


   		// ПАРАЛЛЕЛЬНОЕ ВОЗДЕЙСТВИЕ ОПЕРАТОРА.

   		// y-axis
		MPI_Irecv(&residuals[y_end][x_begin], x_end - x_begin, MPI_FLOAT, top_id, 2001, MPI_COMM_WORLD, &req[0]);
		MPI_Irecv(&residuals[y_begin-1][x_begin], x_end - x_begin, MPI_FLOAT, bot_id, 2002, MPI_COMM_WORLD, &req[1]);
		MPI_Isend(&residuals[y_end-1][x_begin], x_end - x_begin, MPI_FLOAT, top_id, 2002, MPI_COMM_WORLD, &req[2]);	
		MPI_Isend(&residuals[y_begin][x_begin], x_end - x_begin, MPI_FLOAT, bot_id, 2001, MPI_COMM_WORLD, &req[3]);	
		MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
		
		// x-axis
		posleft=0, posright=0;

		MPI_Pack_size(y_end - y_begin, MPI_FLOAT, MPI_COMM_WORLD, &bufsize);
		sbufleft = (float*) malloc(bufsize);
		sbufright = (float*) malloc(bufsize);
		rbufleft = (float*) malloc(bufsize);
		rbufright = (float*) malloc(bufsize);

		/* Pack the data before sending */
		for (int i=y_begin; i<y_end; i++) {
			MPI_Pack(&residuals[i][x_end-1], 1, MPI_FLOAT, sbufright, bufsize, &posright, MPI_COMM_WORLD);
			MPI_Pack(&residuals[i][x_begin], 1, MPI_FLOAT, sbufleft, bufsize, &posleft, MPI_COMM_WORLD);
		}

		/* Execute now the real communication */
		MPI_Irecv(rbufleft, bufsize, MPI_PACKED, left_id, 2001, MPI_COMM_WORLD, &req[0]);
		MPI_Irecv(rbufright, bufsize, MPI_PACKED, right_id, 2002, MPI_COMM_WORLD, &req[1]);
		MPI_Isend(sbufleft, posleft, MPI_PACKED, left_id, 2002, MPI_COMM_WORLD, &req[2]);
		MPI_Isend(sbufright, posright, MPI_PACKED, right_id, 2001, MPI_COMM_WORLD, &req[3]);
		MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

		/* Unpack the received data */
		posright = posleft = 0;
		for (int i=y_begin; i<y_end; i++) {
			if (right_id != MPI_PROC_NULL)
				MPI_Unpack(rbufright, bufsize, &posright, &residuals[i][x_end], 1, MPI_FLOAT, MPI_COMM_WORLD);
			if (left_id != MPI_PROC_NULL)
				MPI_Unpack(rbufleft, bufsize, &posleft, &residuals[i][x_begin-1], 1, MPI_FLOAT, MPI_COMM_WORLD);
		}

		for (int i=y_begin; i<y_end; i++) {
			for (int j=x_begin; j<x_end; j++) {
				float x = A1 + j*h1;
				float y = B1 + i*h2;

				Ar[i][j] = (1.0 / h1 * ((4.0 + x + 0.5*h1)*(residuals[i][j+1] - residuals[i][j]) / h1 - \
										(4.0 + x - 0.5*h1)*(residuals[i][j] - residuals[i][j-1]) / h1) + \
							1.0 / h2 * ((4.0 + x)*(residuals[i+1][j] - 2.0*residuals[i][j] + residuals[i-1][j]) / h2)) * (-1);
				if (x + y > 0.0)
					Ar[i][j] += powf(x + y, 2)*residuals[i][j];			  				  
			}
		}

		free(rbufleft);
		free(sbufleft);
		free(rbufright);
		free(sbufright);

   		// параллельное вычисление скалярного произведения и нормы.

		float local_result_1 = 0.0;
		for (int i=y_begin; i<y_end; i++) {
			for (int j=x_begin; j<x_end; j++) {
				local_result_1 += Ar[i][j] * residuals[i][j] * ro[i][j];
			}	
		}

		local_result_1 *= h1*h2;
		float result_1;
  		MPI_Allreduce(&local_result_1, &result_1, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);


  		float local_result_2 = 0.0;
		for (int i=y_begin; i<y_end; i++) {
			for (int j=x_begin; j<x_end; j++) {
				local_result_2 += Ar[i][j] * Ar[i][j] * ro[i][j];
			}	
		}
		local_result_2 *= h1*h2;
		float result_2;
  		MPI_Allreduce(&local_result_2, &result_2, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

  		tau = result_1 / result_2;	

  		// параллельная работа в цикле: шаг, разность и переприсваивание.

		for (int i=y_begin; i<y_end; i++) {
			for (int j=x_begin; j<x_end; j++) {
				w_new[i][j] = w[i][j] - tau*residuals[i][j];
				diff[i][j] = fabs(w_new[i][j] - w[i][j]);
				w[i][j] = w_new[i][j];
			}
		}

		//if (my_id == 0)
		//	print_array(residuals);
   		// параллельное вычисление скалярного произведения.


   		float local_result_3 = 0.0;
		for (int i=y_begin; i<y_end; i++) {
			for (int j=x_begin; j<x_end; j++) {
				local_result_3 += diff[i][j] * diff[i][j] * ro[i][j];
			}	
		}
		local_result_3 *= h1*h2;
		local_result_3 = powf(local_result_3, 0.5);
		float result_3;
  		MPI_Allreduce(&local_result_3, &result_3, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

  		//if (my_id == 0)
  		//	printf("%f\n", result_3);

  		// free(t);

  		if (result_3 < EPS)
  			error = false;
   	}

   	MPI_Finalize();

   	time_t t1 = time(0);
	double time_in_seconds = difftime(t1, t0);

	printf("TIME IS sf\n", time_in_seconds);

	return 0;
}