/*
 * This program calculates all-to-all kNN of a set X that is given by a .csv as an argument.
 * This version uses mpi to "split" the workload to processes that are connected in a ring topology.
 * An assumption is made, for learning purposes, that the X input set cannot be read by every process.
 * For this reason the X is split as corpus set to the number of processes and then it's passed accordingly to each process by the root process.
 * Afterwards each process passes it's local corpus set to the next and the receiving proccess calculates the kNNs for the local set from the set that was received.
 * At the end each procces has a copy of the knnresult struct that contains the all-to-all kNNs after a simple merginf is done.
 */

#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>
#include<errno.h>
#include<string.h>
#include<math.h>
#include<float.h>
#include<cblas.h>
#include<sys/time.h>
#include "mpi.h"

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
knnresult kNN(double * X, double * Y, int n, int m, int d, int k, int nIndex);
knnresult distrAllkNN(double * X, int n, int d, int k);
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
	/* kNN variables*/
	int n = dataRows;
	int m = dataRows;
	int d = attributes;
	int k = 10;
	double *X = (double *)malloc(n*d*sizeof(double));
	/* Init X and Y */
	for (int i= 0 ; i<n; i++){
		for (int j=0; j<d ; j++){
			X[d*i+j]=data[i][j];
		}
	}

	/* Free  data memory*/
	while(dataRows) free(data[--dataRows]);
	free(data);
	/* Variables for time measurment  */
	struct timeval timeStart,timeEnd;
	double totaltime;
	/* kNN Calculation */
	gettimeofday(&timeStart,NULL);
	knnresult res=distrAllkNN(X,n,d,k);
	gettimeofday(&timeEnd,NULL);
	totaltime = (timeEnd.tv_sec*1000 + timeEnd.tv_usec/1000) - (timeStart.tv_sec*1000 + timeStart.tv_usec/1000) ;
	printf("Total time : %.4f \n",totaltime);

	//for (int i = 0 ; i<m*k;i++)
	//	printf("dist: %lf \t index: %d  \n",res.ndist[i],res.nidx[i]);
	//
	/*Ending Proccess Memory Dealloc*/
	free(X);
	fclose(fp);
	exit(EXIT_SUCCESS);

}

knnresult
distrAllkNN(double * X, int n, int d, int k){

	/*
	 * In order to balance the processes 
	 * Tag = 0 -> corpus[m] : m = (n/p) + 1
	 * Tag = 1 -> corpus[m] : m = (n/p) 
	 *
	 * TagIdx = (n % numtasks)
	 * Tag = rank < TagIdx ? 0 : 1
	 */

	#define MASTER 0

	/* MPI variables declaration */
	int numtasks,rank,next,previous,Tag,TagIdx,m,mIn,nCorpus;
	double *Corpus,*Y,*Z;	

	MPI_Status Stat;

	MPI_Init(NULL,NULL);
	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	int displacements[numtasks];
	TagIdx = n % numtasks;
	/* Master <- Partition X and send it to Slace processes  */
	if (rank == MASTER){
		int nBlock;
		/* Find Master's X block size  */
		if(0 < TagIdx)
			nCorpus = floor(n/numtasks) +1;
		else
			nCorpus = floor(n/numtasks);
		/* Iterate through X's Blocks and send them accordingly to processes */
		int XIdx=nCorpus;
		displacements[0]=0;
		for (int i=1; i<numtasks;i++){
			/* Find block size */
			if (i < TagIdx){
				nBlock = floor(n/numtasks) +1;
			}
			else {
				nBlock = floor(n/numtasks);
			}
			/* Send */
			MPI_Send(&X[d * (XIdx)],nBlock * d,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
			/* Fill the displacements array for later use */ 
			displacements[i]=XIdx*k;
			/* Add the blocksize to the Xindex in order to know the next starting address */
			XIdx+=nBlock;
		}
		/* Assign the first X block to master's corpus and Y*/
		Corpus 	= (double *)malloc(nCorpus*d*sizeof(double));
		Y	= (double *)malloc(nCorpus*d*sizeof(double));
		for (int i=0; i<nCorpus*d;i++){
			Corpus[i]=X[i];
			Y[i]	 =X[i];
		}
		/* Update 1st m */
		m=nCorpus;

	}

	/* Slaves <- Receive X partioned (Corpus set) */
	if (rank != MASTER){
		MPI_Status InitialReceiveStatus;
		/* Find Expected Block Size */
		if (rank<TagIdx){
			nCorpus = floor(n/numtasks) +1;
		}
		else{
			nCorpus = floor(n/numtasks);
		}
		Corpus 	= (double *)malloc(nCorpus*d*sizeof(double));
		Y	= (double *)malloc(nCorpus*d*sizeof(double));
		/* Receive Corpus */
		MPI_Recv(Corpus,nCorpus * d,MPI_DOUBLE,MASTER,0,MPI_COMM_WORLD,&InitialReceiveStatus);
		/* Pass it to Y */
		for (int i=0; i<nCorpus*d;i++){
			Y[i]	 =Corpus[i];
		}
		m=nCorpus;
		
	}	
	/* Assign Destinations and Sources  */
	next = rank + 1;
	previous =rank - 1;
	if (rank == numtasks-1) next = 0;
	if (rank == 0) previous = numtasks - 1;
	/* Find the first Tag (see below function's definition) */
	Tag = rank < TagIdx ? 0 : 1 ;
	/* Start Iteration  */
	MPI_Request reqs[2];
	MPI_Status stat[2],TagStat;
	knnresult result;
	MPI_Bcast(displacements,numtasks,MPI_INT,MASTER,MPI_COMM_WORLD);
	int Ycurrent = rank;
	/* Start Iteration  */
	for (int iteration = 0 ; iteration < numtasks ; iteration++){
		/* For the last iteration don't send anything */
		if (iteration < numtasks -1){
			/* Send Y to neighbours */
			MPI_Isend(Y,m*d,MPI_DOUBLE,next,Tag,MPI_COMM_WORLD,&reqs[0]);
			/* Probe for the Tag to get the incoming dimensions */
			MPI_Probe(previous,MPI_ANY_TAG,MPI_COMM_WORLD,&TagStat);
			Tag = TagStat.MPI_TAG;
			/* Allocate memory for the incoming */
			mIn = Tag == 0 ? floor(n/numtasks) +1 : floor(n/numtasks);
			Z = (double *)malloc(mIn*d*sizeof(double));
			/* Non-Blocking Receive */
			MPI_Irecv(Z,mIn*d,MPI_DOUBLE,previous,Tag,MPI_COMM_WORLD,&reqs[1]); 
		}
		

		/* Calculate kNNs for Y */
		result = kNN(Y,Corpus,m,nCorpus,d,k,(Ycurrent == 0 )? 0 :displacements[Ycurrent]/k);
		/* If is the last iteration skip wait */
		if (iteration < numtasks -1){
			/* Wait for the asynchrous send-receive to return */
			MPI_Waitall(2,reqs,stat);
			/* Free Y and swap Y and Z */
			free(Y);
			Y=Z;
			m=mIn;
			Ycurrent--;
			if (Ycurrent == -1) Ycurrent = numtasks - 1 ;
		}
		

	}
	/* Sort the result */
	double 	*finalDist = (double *)malloc(n*k*sizeof(double));
	int 	*finalIdx = (int *)malloc(n*k*sizeof(int));
	int receiveCount[numtasks];
	int countCorpus = nCorpus*k;
	MPI_Allgather(&countCorpus,1,MPI_INT,receiveCount,1,MPI_INT,MPI_COMM_WORLD);


	MPI_Allgatherv(result.ndist,nCorpus*k,MPI_DOUBLE,finalDist,receiveCount,displacements,MPI_DOUBLE,MPI_COMM_WORLD);
	MPI_Allgatherv(result.nidx ,nCorpus*k,MPI_INT   ,finalIdx ,receiveCount,displacements,MPI_INT   ,MPI_COMM_WORLD);
	/* Deallocate Z */
	free(Z);
	MPI_Finalize();
	
	/* Setup result for output */
	result.k=k;
	result.m=n;

	return result;

}

knnresult
kNN(double * X, double * Y, int n, int m, int d, int k, int nIndex){
	# define numOfBlocks 120
	/* knnresult struct variables */
	static int    *idx     = NULL;
	static double *dist    = NULL;
	static int iter = 0;
	iter++;
	if (idx==NULL){
		idx=(int *)   malloc(m*k*sizeof(int));
		dist = (double *)malloc(m*k*sizeof(double));
		/* Initialize dist and idx */
		for (int i=0;i<m*k;i++){
			dist[i]=DBL_MAX;
			idx[i]=-1;
		}
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
		
		/* Find the k nearest neighbours */
		for (int j=0;j<n;j++){
			for (int i=0;i<mBlock;i++){
				/* Check if D is less than dist for every k  */
				for (int knn=k-1;knn>=0;knn--){
					/* Exclude 0 Values */
					if (D[j*mBlock+i] <= dist[(i+mIndex)*k+knn] &&  D[j*mBlock+i] > 0.0001){
						double tmpdist = dist[(i+mIndex)*k+knn];
						int tmpidx = idx[(i+mIndex)*k+knn];
						dist[(i+mIndex)*k+knn] = D[j*mBlock+i] ;
						idx[(i+mIndex)*k+knn] = j + nIndex ;
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
	/* Get first line from file */
	/* and check if file is empty */
	if (fgets(buf,2048,fp)==NULL){
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
