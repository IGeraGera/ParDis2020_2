#include<stdio.h>
#include<stdlib.h>
#include<errno.h>
#include<string.h>
#include<math.h>
#include<cblas.h>

// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

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
//knnresult kNN(double * X, double * Y, int n, int m, int d, int k);
void checkArgsNum(int argc);
int getAttributes(FILE *fp);
double ** readCSV(FILE *fp,double **data, int *dataRows);

extern int errno ; 
int
main(int argc, char *argv[]){
	checkArgsNum(argc);
	FILE *fp;
	/* Open File and Handle errors */
	fp= fopen (argv[1],"r");
	if (fp == NULL){
		fprintf(stderr, "Line %d: Error opening file %s\n",__LINE__,strerror(errno));
		exit (EXIT_FAILURE);
	}
	/* Read csv */
	double **data= (double **)malloc(0);
	int dataRows;
	data=readCSV(fp,data,&dataRows);
	int attributes = getAttributes(fp);
	/* Print data */
	/*
	for(int i=0; i<dataRows;i++){
		for(int j = 0 ; j<attributes ;j++){
			printf("%lf ",data[i][j]);
		}
		printf("\n");
	}
	*/
	/* kNN variables*/
	# define splitcoeff 0.5
	int n = dataRows*splitcoeff;
	int m = dataRows - n;
	int d = attributes;
	int k = 3;
	double *X = (double *)malloc(n*d*sizeof(double));
	double *Y = (double *)malloc(m*d*sizeof(double));
	/* Init X and Y */
	for (int i= 0 ; i<n; i++){
		for (int j=0; j<d ; j++){
			X[d*i+j]=data[i][j];
		}
	}
	for (int i= 0 ; i<m; i++){
		for (int j=0; j<d ; j++){
			Y[d*i+j]=data[i+n][j];
		}
	}
	/* Starting Procces */
	double *X_HAD   = (double *)malloc(n*d*sizeof(double));
	double *Y_HAD   = (double *)malloc(m*d*sizeof(double));
	double *e_d_yt  = (double *)malloc(d*m*sizeof(double));
	double *e_x_dt  = (double *)malloc(n*d*sizeof(double));
	double *X_part  = (double *)malloc(n*m*sizeof(double));
	double *Y_part  = (double *)malloc(n*m*sizeof(double));
	double *XY_part = (double *)malloc(n*m*sizeof(double));
	double *D       = (double *)malloc(n*m*sizeof(double));
	int    *idx     = (int *)   malloc(m*k*sizeof(int));
	double *dist    = (double *)malloc(m*k*sizeof(double));
	struct knnresult result;

	/* Init dist */
	for (int i=0;i<m*k;i++){
		dist[i]=0;
		idx[i]=-1;
	}

	/* X Hadamart */
	for (int i=0;i<n*d;i++){
		X_HAD[i]=X[i]*X[i];
		e_x_dt[i]=1.0;
	}
	/* X Hadamart */
	for (int i=0;i<m*d;i++){
		Y_HAD[i]=Y[i]*Y[i];
		e_d_yt[i]=1.0;
	}
	/* Main CBLAS part */
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,m,d,1.0,X_HAD,d,e_d_yt,m,0,X_part,m);
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,m,d,1.0,e_x_dt,d,Y_HAD,d,0,Y_part,m);
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,m,d,2.0,X,d,Y,d,0,XY_part,m);
	/* D = sqrt(Xpart - XYpart - Ypart) */	
	for (int i=0;i<n*m;i++){
		D[i]=sqrt(X_part[i] - XY_part[i]  + Y_part[i]);
	}
	/* Prints D  */
	/*
	for (int i= 0 ; i<n; i++){
		for (int j=0; j<m ; j++){
			printf("%lf ",D[m*i+j]);
		}
		puts("");
	}
	puts("");
	*/	
	/* Find the k nearest neighbours */
	for (int j=0;j<n;j++){
		for (int i=0;i<m;i++){
			
			for (int knn=0;knn<k;knn++){
				if (D[j*m+i] >= dist[i*k+knn]){
					double tmpdist = dist[i*k+knn];
					int tmpidx = idx[i*k+knn];
					dist[i*k+knn] = D[j*m+i] ;
					idx[i*k+knn] = j*m+i;
					/* If k index (knn) is 0 there is no place to put the tmp so break  */
					if (knn!=0){
						dist[i*k+knn-1] = tmpdist;
						idx[i*k+knn-1] = tmpidx;
					}
				}
			}
		}
	}
	/*	
	for (int i = 0 ; i<m*k;i++)
		printf("dist: %lf \t index: %d  \n",dist[i],idx[i]);
	*/


	free(X_HAD);	
	free(Y_HAD);	
	free(e_d_yt);	
	free(e_x_dt);	
	free(X_part);
	free(Y_part);
	free(XY_part);
	free(D);
	/*Ending Proccess  */
	free(X);
	free(Y);
	free(data);
	fclose(fp);
	exit(EXIT_SUCCESS);

}

void
checkArgsNum(int argc){
	switch(argc){
	case 1:
		fprintf(stderr, "Line:%d No argument was given\n",__LINE__);
		exit(EXIT_FAILURE);
	case 2:
		break;
	default:
		fprintf(stderr, "Line:%d More args given\n",__LINE__);
		exit(EXIT_FAILURE);
	}
}
int
getAttributes(FILE *fp){
	/* Variables */
	char buf[2048];
	int attributes=0;
	char delim=',';
	/* Gete first line from file */
	fgets(buf,2048,fp);
	/* check if file is empty */
	if (buf==NULL){
		fprintf(stderr,"Line %d: Input file empty\n",__LINE__ );
		exit(EXIT_FAILURE);
	}
	/* Count the occurences of the delimiters */
	for(int i=0;i<strlen(buf);i++){
		if (buf[i]==delim) attributes++;
	}
	/* Add 1 to the occurrences of the delimiter */
	attributes++;
	/* Rewind fp */
	rewind(fp);
	return attributes;
}

double **
readCSV(FILE *fp, double **data, int *dataRows){
	/* Variables */
	int attributes = getAttributes(fp);
	const char delim[] = ",";
	int row=0;
	char buf[2048];
	/* Read Lines one by one and split them on the delimiter */
	while(fgets(buf,2048,fp)){
		/* realloc the data array to fill another row */
		data = (double **)realloc(data,(row+1)*sizeof(double *));
		data[row] = (double *)malloc(attributes*(sizeof(double)));
		/* Split the buf on the delimiter and fill the row */
		char *token;
		for (int i = 0; i< attributes; i++){
			if (i==0)
				token = strtok(buf,delim); 
			else
				token = strtok(NULL,delim);
			/* If token NULL no more lines exist (Maybe there is no need for this) */
			if (token==NULL) break;
			/* Covert str to double */
			sscanf(token, "%lf",&data[row][i]);
		}
		row++;
	}
	/* Return dataRows */
	*dataRows=row;
	rewind(fp);
	return data;
}
