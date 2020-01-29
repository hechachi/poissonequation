#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
// количество строк. 
const int N = 100;
// количество столбцов.
const int M = 100;

// область определения функции.
const int A1 = -1;
const int A2 = 2;
const int B1 = -2;
const int B2 = 2;

const float EPS = 1e-6;
const float h1 = float((A2 - A1)) / float(M);
const float h2 = float((B2 - B1)) / float(N);

static float ro[N+1][M+1];
static float B[N+1][M+1];
static float diff[N+1][M+1];
static float function_value[N+1][M+1];
static float w[N+1][M+1] = {0.0};
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
			//cout << arr[N-i][j] << ' ';
		}
		printf("\n");
		//cout << endl;
	}
	printf("\n\n");
	//cout << endl;
};

// скалярное произведение в пространстве H сеточных функций.
float dot_product(float x[N+1][M+1], float y[N+1][M+1]) {
	
	float result = 0.0;
	
	for (int i=1; i<N; i++) {
		for (int j=1; j<M; j++) {
			result += x[i][j] * y[i][j] * ro[i][j];
		}	
	}
	result *= h1*h2;
	return result;
};

// норма в пространстве H сеточных функций.
float norm(float x[N+1][M+1]) {
	return powf(dot_product(x, x), 0.5);
};

// процедура, тождественно равная действию оператора A.
void operator_A(float w[N+1][M+1], float w_new[N+1][M+1]) {

	for (int i=1; i<N; i++) {
		for (int j=1; j<M; j++) {
			float x = A1 + j*h1;
			float y = B1 + i*h2;

			w_new[i][j] = (1.0 / h1 * ((4.0 + x + 0.5*h1)*(w[i][j+1] - w[i][j]) / h1 - \
									   (4.0 + x - 0.5*h1)*(w[i][j] - w[i][j-1]) / h1) + \
						   1.0 / h2 * ((4.0 + x)*(w[i+1][j] - 2.0*w[i][j] + w[i-1][j]) / h2)) * (-1);
			if (x + y > 0.0)
				w_new[i][j] += powf(x + y, 2)*w[i][j];			  				  
		}
	}
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


int main() {

	bool error = true;
	float tau;
	float diff_norm;

	// получаем значения искомой функции на сетке.
	for (int i=0; i<N+1; i++) {
		for (int j=0; j<M+1; j++) {
			float x = A1 + j*h1;
			float y = B1 + i*h2;
			function_value[i][j] = exp(1 - pow(x+y, 2));
		}
	}

	//print_array(function_value);		
	
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


	// основной цикл программы.
	while (error) {

		float Aw[N+1][M+1] = {0.0};
		float Ar[N+1][M+1] = {0.0};

		boundary_init(Aw);

		operator_A(w, Aw);

		for (int i=0; i<N+1; i++) {
			for (int j=0; j<M+1; j++) {
				residuals[i][j] = Aw[i][j] - B[i][j];
			}
		}

		//print_array(residuals);

		operator_A(residuals, Ar);

		tau = dot_product(Ar, residuals) / pow(norm(Ar), 2);

		for (int i=0; i<N+1; i++) {
			for (int j=0; j<M+1; j++) {
				w_new[i][j] = w[i][j] - tau*residuals[i][j];
				diff[i][j] = fabs(w_new[i][j] - w[i][j]);
				w[i][j] = w_new[i][j];
			}
		}

		diff_norm = norm(diff);

		printf("%f\n", diff_norm);

		if (diff_norm < EPS)
			error = false;	
	}

		// запись в файл.
	FILE *qfile;
	qfile=fopen("values.txt", "wb");

	for (int i=0; i<N+1; i++) {
		for (int j=0; j<M+1; j++) {
    		fprintf(qfile, "%f ", w[i][j]);	
		}
		fprintf(qfile,"\n");	
	}

	float estimation = estimation_of_accuracy(function_value, w);
	printf("estimation = %f\n", estimation);

	float difference[N+1][M+1];
	for (int i=0; i<N; i++) {
		for (int j=0; j<M; j++) {
			difference[i][j] = w[i][j] - function_value[i][j];
		}
	}

	print_array(difference);

	return 0;
}