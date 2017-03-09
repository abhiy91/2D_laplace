/*
Parallel implementation of the 2D Laplace equation using the Conjugate Gradient method and Compressed Row Storage for sparse matrices
*/

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<mpi.h>

double L=1.0;
double W=1.0;
int max=500;
int itrmax=100000;
double pi=3.14159265359;
double dx=L/(double)max;
double dy=W/(double)max;
double d=dx/dy;
double tol=0.000001;
int ROOT=0;


int main(int argc, char *argv[]){
	int i,j,k,l;
	double *b_v, *x_sol, **x_new_m, *x_new, *p, *p_0, *r;
	double *x, *y;
	int allpts;
	int ta,tb,t;
	int itr=0,itrcount=0;
	double error, sum, sum_rank;
	double alpha=0, alpha_num=0, alpha_num_rank=0, alpha_den=0, alpha_den_rank=0, beta=0, beta_num=0, beta_num_rank=0;
	double *temp,*z;
	int nprocs,myid,ierr;
	int start, end, prows;
	MPI_Status *status;
	double t_start, t_end;
	
	ierr = MPI_Init(&argc,&argv);
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	if(myid==ROOT){
	  t_start = MPI_Wtime();
	}

	//distribute A matrix by rows
	//calculate the number of rows, start point and end point for each process
	int rcounts[nprocs];
	int displ[nprocs];

	allpts = (max-1)*(max-1);

	int rem = (allpts%nprocs);
	int div = (allpts/nprocs);

	if(myid<rem){
	  prows = div + 1;
	}else{
	  prows = div;
	}

	if(myid==ROOT){
	  start = 0;
	}else if((myid-1)<rem){
	  start = ((myid * div) + myid);
	}else{
	  start = ((myid * div) + rem);
	}

	end = start + prows - 1;

	printf("id=%d start=%d end=%d prows=%d\n",myid,start,end,prows);

	//populate rcounts and displ
	for(i=0;i<nprocs;i++){
	  if(i<rem){
	    rcounts[i] = div + 1;
	  }else{
	    rcounts[i] = div;
	  }

	  if(i==0){
	    displ[i] = 0;
	  }else if((i-1)<rem){
	    displ[i] = i*div + i;
	  }else{
	    displ[i] = i*div + rem;
	  }
	}

	//memory allocation
	x = (double*)malloc((max+1)*sizeof(double));
	y = (double*)malloc((max+1)*sizeof(double));

	if(myid==ROOT){
	  x_new = (double*)malloc((allpts)*sizeof(double));
	  x_new_m = (double**)malloc((max-1)*sizeof(double*));
	  for(i=0;i<max-1;i++){
	    x_new_m[i] = &x_new[i*(max-1)];
	  }
	}

 	b_v = (double*)malloc((prows)*sizeof(double));
 	x_sol = (double*)malloc((prows)*sizeof(double));
 	p = (double*)malloc((prows)*sizeof(double));
	p_0 = (double*)malloc((allpts)*sizeof(double));
 	r = (double*)malloc((prows)*sizeof(double));
 	temp = (double*)malloc((prows)*sizeof(double));
 	z = (double*)malloc((prows)*sizeof(double));

	if(myid==ROOT){
	  //generate grid
	  for(i=0;i<=max;i++){
		x[i] = dx*i;
	  }
	  for(j=0;j<=max;j++){
		y[j] = dy*j;
	  }
	}
	
	//Counter for non-zero elements in A
	int acount = 0;

	for(t=start;t<=end;t++){
	  acount += 1;
	  if(t%(max-1)!=0){
	    acount += 1;
	  }
	  if(t%(max-1)!=(max-2)){
	    acount += 1;
	  }
	  if(t>(max-2)){
	    acount += 1;
	  }
	  if(t<(allpts-(max-1))){
	    acount += 1;
	  }
	}

	//define variables for CRS
	double *A_val;
	int *A_colind, *A_rowptr;
	int valcount=0;
	int A;
	
	//memory allocation for CRS
	A_val = (double*)malloc(acount*sizeof(double));
	A_colind = (int*)malloc(acount*sizeof(int));
	A_rowptr = (int*)malloc((prows+1)*sizeof(int));

	//store elements of matrix A in CRS format
	for(t=start;t<=end;t++){
		A_rowptr[t-start] = valcount;
		A_val[valcount] = -2*(1+d*d);
		A_colind[valcount] = t;
		valcount++;
		if(t%(max-1)<max-2){
			A_val[valcount] = 1;
			A_colind[valcount] = t+1;
			valcount++;
		}
		if(t%(max-1)>0 && t%(max-1)<(max-1)){
			A_val[valcount] = 1;
			A_colind[valcount] = t-1;
			valcount++;
		}
		if(t>(max-2)){
			A_val[valcount] = d*d;
			A_colind[valcount] = t-(max-1);
			valcount++;
		}
		if(t<(allpts-(max-1))){
			A_val[valcount] = d*d;
			A_colind[valcount] = t+(max-1);
			valcount++;
		}
		if(t==end){
			A_rowptr[end-start+1] = valcount;
		}
	}

	//polpulate b vector
	for(t=start;t<=end;t++){
	  b_v[t-start] = 0;
	  if(t%(max-1)==0){
	    b_v[t-start] -= 1*0;
	  }
	  if(t%(max-1)==(max-2)){
	    b_v[t-start] -= 1*0;
	  }
	  if(t<(max-1)){
	    b_v[t-start] -= (d*d)*sin(pi*x[t%(max-1)+1]);
	  }
	  if(t>(allpts-max)){
	    b_v[t-start] -= (d*d)*sin(pi*x[t%(max-1)+1])*exp(-1*pi);
	  }
	}
	
	for(i=start;i<=end;i++){
		x_sol[i-start] = 0;
		r[i-start] = b_v[i-start];
		//z = r*M_inv preconditioned
		z[i-start] = r[i-start]*(1/A_val[A_rowptr[i-start]]);
		p[i-start] = z[i-start];
	}

	error = 100;
	//conjugate gradient method
	while(error>tol && itr<itrmax){
		itr++;
		
		//initialize temp
		for(i=start;i<=end;i++){
			temp[i-start] = 0;
		}

		//gather data for p_old from all processes
		ierr = MPI_Allgatherv(p,prows,MPI_DOUBLE,p_0,rcounts,displ,MPI_DOUBLE,MPI_COMM_WORLD);

		//A*p_old
		for(i=start;i<=end;i++){
			for(k=A_rowptr[i-start];k<A_rowptr[i-start+1];k++){
				temp[i-start] += A_val[k]*p_0[A_colind[k]];
			}
		}

		//alpha = r_old'*z_old / p_old*A*p_old
		alpha_den_rank = 0;
		for(i=start;i<=end;i++){
			alpha_den_rank += temp[i-start]*p_0[i];
		}
		
		alpha_den = 0;
		ierr = MPI_Allreduce(&alpha_den_rank,&alpha_den,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		
		alpha_num_rank = 0;
		for(i=start;i<=end;i++){
			alpha_num_rank += r[i-start] * z[i-start];
		}
		
		alpha_num = 0;
		ierr = MPI_Allreduce(&alpha_num_rank,&alpha_num,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

		alpha = alpha_num/alpha_den;
		
		//x_new = x_old + alpha * p_old
		for(i=start;i<=end;i++){
			x_sol[i-start] = x_sol[i-start] + alpha*p[i-start];
		}
		
		//r_new = r_old - alpha*A*p_old
		//z_new = M_inv*r_new
		for(i=start;i<=end;i++){
			r[i-start] = r[i-start] - alpha*temp[i-start];
			z[i-start] = r[i-start]*(1/A_val[A_rowptr[i-start]]);
		}
		
		//beta = z_new' * r_new / z_old' * r_old
		beta_num_rank = 0;
		for(i=start;i<=end;i++){
			beta_num_rank += z[i-start] * r[i-start];
		}

		ierr = MPI_Allreduce(&beta_num_rank,&beta_num,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		
		beta = beta_num/alpha_num;
		
		//p_new = z_new + beta*p_old
		for(i=start;i<=end;i++){
			p[i-start] = z[i-start] + beta*p[i-start];
		}
		
		//calculate the error to check for convergence
		sum=0;
		sum_rank=0;

		for(i=start;i<=end;i++){
		  sum_rank += r[i-start] * r[i-start];
		}
		MPI_Allreduce(&sum_rank,&sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		error = sqrt(sum);
	}

	//gather final solution on root processor for printing
	MPI_Gatherv(x_sol,prows,MPI_DOUBLE,x_new,rcounts,displ,MPI_DOUBLE,ROOT,MPI_COMM_WORLD);

	if(myid==ROOT){
	  t_end = MPI_Wtime();

	  printf("time = %.09lf\n iterations = %d\n", (t_end-t_start), itr);
	}

 	//free memory
	free(x);
	free(y);

	if(myid==ROOT){
	  free(x_new);
	  free(x_new_m);
	}

	free(b_v);
	free(p);
	free(p_0);
	free(r);
	free(x_sol);
	free(z);
	free(temp);
		
	ierr=MPI_Finalize();	
	return 0;
}


	
