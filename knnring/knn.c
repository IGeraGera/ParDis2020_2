#include<stdio.h>
#include<stdlib.h>
#include<errno.h>
#include<string.h>
#include<math.h>
#include<cblas.h>
#include<float.h>
#include "inc/knnring.h"

//! Compute k nearest neighbors of each point in X [n-by-d]
/*!

  \param  X      Corpus data points              [n-by-d]
  \param  Y      Query data points               [m-by-d]
  \param  n      Number of corpus points         [scalar]
  \param  m      Number of query points          [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/

/* Functions Declaration */

knnresult kNN(double * X, double * Y, int n, int m, int d, int k){
	# define numOfBlocks 8
	/* knnresult struct variables */
	int    *idx     = (int *)   malloc(m*k*sizeof(int));
	double *dist    = (double *)malloc(m*k*sizeof(double));
	/* Initialize dist and idx */
	for (int i=0;i<m*k;i++){
		dist[i]=DBL_MAX;
		idx[i]=-1;
	}
	/* Variables that only allocate once */
	double *X_HAD   = (double *)malloc(n*d*sizeof(double));
	double *e_x_dt  = (double *)malloc(n*d*sizeof(double));
	/* Initialize X Hadamart and e_x_dt */
	for (int i=0;i<n*d;i++){
		X_HAD[i]=X[i]*X[i];
		e_x_dt[i]=1.0;
	}
	/* Block the Y */
	for (int current_block=1; current_block <= numOfBlocks; current_block++){
		/* Block variables */
		int mBlock = m / numOfBlocks;
		int mIndex = mBlock * (current_block-1);
		if (current_block == numOfBlocks)
			mBlock= m - (numOfBlocks-1) * mBlock;
		/* Main variables */
		double *Y_HAD   = (double *)malloc(mBlock*d*sizeof(double));
		double *e_d_yt  = (double *)malloc(d*mBlock*sizeof(double));
		double *X_part  = (double *)malloc(n*mBlock*sizeof(double));
		double *Y_part  = (double *)malloc(n*mBlock*sizeof(double));
		double *XY_part = (double *)malloc(n*mBlock*sizeof(double));
		double *D       = (double *)malloc(n*mBlock*sizeof(double));
		/* Initialize Y Hadamart */
		for (int i=0;i<mBlock*d;i++){
			Y_HAD[i]=Y[mIndex*d+i]*Y[mIndex*d+i];
			e_d_yt[i]=1.0;
		}
		/* Main CBLAS part */
		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,mBlock,d,1.0,X_HAD,d,e_d_yt,mBlock,0,X_part,mBlock);
		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,mBlock,d,1.0,e_x_dt,d,Y_HAD,d,0,Y_part,mBlock);
		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,mBlock,d,2.0,X,d,&Y[mIndex*d],d,0,XY_part,mBlock);
		/* D = sqrt(Xpart - XYpart - Ypart) */	
		for (int i=0;i<n*mBlock;i++){
			D[i]=sqrt(X_part[i] - XY_part[i]  + Y_part[i]);
		}
		/* Prints D  */
		/*
		for (int i= 0 ; i<n; i++){
			for (int j=0; j<mBlock ; j++){
				printf("%lf ",D[mBlock*i+j]);
			}
			puts("");
		}
		puts("");
		*/

		/* Find the k nearest neighbours */
		for (int j=0;j<n;j++){
			for (int i=0;i<mBlock;i++){
				/* Check if D is greater than dist for every k  */
				for (int knn=k-1;knn>=0;knn--){
					if (D[j*mBlock+i] <= dist[(i+mIndex)*k+knn]){
						double tmpdist = dist[(i+mIndex)*k+knn];
						int tmpidx = idx[(i+mIndex)*k+knn];
						dist[(i+mIndex)*k+knn] = D[j*mBlock+i] ;
						idx[(i+mIndex)*k+knn] = j;
						/* If k index (knn) is 0 there is no place to put the tmp so break  */
						if (knn!=k-1){
							dist[(i+mIndex)*k+knn+1] = tmpdist;
							idx[(i+mIndex)*k+knn+1] = tmpidx;
						}
					}
				}
			}
		}
		/* Free Block variables */
		free(X_part);
		free(Y_part);
		free(XY_part);
		free(e_d_yt);	
		free(Y_HAD);	
		free(D);

		//current_block++;
	}
	/* Prinf dist and idx  */
	/*
	for (int i = 0 ; i<m*k;i++)
		printf("dist: %lf \t index: %d  \n",dist[i],idx[i]);
	*/
	/* Set up knnresult  */
	knnresult result;
	result.m = m;
	result.k = k;
	result.ndist = dist;
	result.nidx = idx;

	free(X_HAD);	
	free(e_x_dt);	

	return result;
}
