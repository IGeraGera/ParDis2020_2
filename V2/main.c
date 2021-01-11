#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>
#include<errno.h>
#include<string.h>
#include<math.h>
#include<float.h>
#include<limits.h>
#include<time.h>
#include<sys/time.h>
#include "mpi.h"

// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

// Definition the tree return
typedef struct treeReturn{
  int currNode;
  int *treeIdx;
  int *treeData;
  double *muArray;
}treeReturn;
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
knnresult distrAllkNN(double * X, int n, int d, int k);
knnresult kNNSearchTree(double *X,int d,treeReturn tree,int B,knnresult result,double *Y,int YIdx,int IdxStart);
knnresult checkIfNN(double dist,int elemIndex,knnresult result,int IdxStart);
treeReturn makeTree(double *X,int n,int d, int *treeIdx,double *muArray, int *treeData,int *treeDataSize,int *availableIndices,int availableIndicesSize,const int B,int *nodes);
int selectVp(double *X,int n,int d,int *availableIndices,int availableIndicesSize);
double calcDistance(double *X,double *Y, int d,int AIdx,int BIdx);
void quicksort(double *distance,int * idxArray,int first,int last);
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
	int n = dataRows;
	int m = dataRows;
	int d = attributes;
	int k = 3;
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

//	for (int i = 0 ; i<m*k;i++)
//		printf("dist: %lf \t index: %d  \n",res.ndist[i],res.nidx[i]);
	
	/*Ending Proccess Memory Dealloc*/
	free(X);
	fclose(fp);
	exit(EXIT_SUCCESS);

}

knnresult
distrAllkNN(double * X, int n, int d, int k){


	#define MASTER 0

	/* MPI variables declaration */
	int numtasks,rank,next,previous,TagIdx,m,nCorpus;
	double *Corpus,*Y;	

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
	/* Prints Corpus  */
	/*
	for (int i= 0 ; i<nCorpus; i++){
		for (int j=0; j<d ; j++){
			printf("%lf ",Y[d*i+j]);
		}
		puts("");
	}
	puts("");
	*/
	/* Make the tree  */
	/* Tree Indices  */
	/* [ vpIdx left right | vpIdx left right .... | vpIdx left right ]*/
	#define vpIdx 0
	#define left 1
	#define right 2
	int *treeIdx = (int *)malloc(0* sizeof(int));
	/* Num of elements in leaves B */
	const int B = 2;
	/* Data array */
	int treeDataSize =1,currNode;
	int *treeData = (int *)malloc(B*sizeof(int));
	for(int i=0;i<B;i++)
		treeData[i]=-1;
	/* Num of Nodes*/
	int nodes=0;
	/* Avilable Indices matrix and Init it*/
	int *availableIndices = (int *)malloc(nCorpus*sizeof(int));
	for (int i=0;i<nCorpus;i++)
		availableIndices[i]=i;
	/* mu array with size [nodes]*/
	double *muArray=NULL;
	/* Make tree
	 * Tree Elememts
	 * treeIdx : contains Vantage point index in X, left node number, right node number.
	 * If the left and right is a leaf then the number that contains is negative that has the index of the data in treeData.
	 * If the leaf is empty it contains the INT_MIN.
	 * muArray : contains radius for inside/outside elemenents
	 * treeData: contains the data of the leaves
	 * availableIndices: INPUT ONLY available indices from the dataset
	 */
	printf("Making tree rank %d\n",rank);
	treeReturn tree = makeTree(Corpus,nCorpus,d,treeIdx,muArray,treeData,&treeDataSize,availableIndices,nCorpus,B,&nodes);
	treeIdx = tree.treeIdx;
	treeData = tree.treeData;
	muArray = tree.muArray;
	currNode = tree.currNode;
	/* Init knnresult struct */
	knnresult result[m];
		for(int y=0;y<m;y++){
			result[y].nidx=(int*)malloc(k*sizeof(int));
			result[y].ndist=(double *)malloc(k*sizeof(double));
			result[y].k=k;
			for(int i=0;i<k;i++){
				result[y].nidx[i]=-1;
				result[y].ndist[i]=DBL_MAX;
			}
		}

	/* Assign Destinations and Sources  */
	next = rank + 1;
	previous =rank - 1;
	if (rank == numtasks-1) next = 0;
	if (rank == 0) previous = numtasks - 1;
	/* Variables buffers  */
	double *Corpusbuf,*muArraybuf;
	int *treeIdxbuf,*treeDatabuf;
	int nodesbuf,nCorpusbuf,treeDataSizebuf,currNodebuf,sendpacket[4],recvpacket[4];
	MPI_Request reqs[8];
	MPI_Status stat[8],packetStat;
	MPI_Bcast(displacements,numtasks,MPI_INT,MASTER,MPI_COMM_WORLD);
	int Ycurrent = rank;
	/* Start Iteration  */
	for (int iteration = 0 ; iteration < numtasks ; iteration++){
		/* For the last iteration don't send anything */
		if (iteration < numtasks -1){
			/* Send the sizes of everything */
			sendpacket[0]=nCorpus;
			sendpacket[1]=nodes;
			sendpacket[2]=treeDataSize;
			sendpacket[3]=currNode;
			MPI_Sendrecv(sendpacket,4,MPI_INT,next,0,recvpacket,4,MPI_INT,previous,0,MPI_COMM_WORLD,&packetStat);
			/* Send arrays to neighbours */
			MPI_Isend(Corpus,nCorpus*d,MPI_DOUBLE,next,0,MPI_COMM_WORLD,&reqs[0]);
			MPI_Isend(muArray,nodes,MPI_DOUBLE,next,0,MPI_COMM_WORLD,&reqs[1]);
			MPI_Isend(treeIdx,nodes*3,MPI_INT,next,0,MPI_COMM_WORLD,&reqs[2]);
			MPI_Isend(treeData,treeDataSize*B,MPI_INT,next,0,MPI_COMM_WORLD,&reqs[3]);
			/* Allocate memory for the incoming */
			nCorpusbuf=recvpacket[0];
			nodesbuf=recvpacket[1];
			treeDataSizebuf=recvpacket[2];
			currNodebuf=recvpacket[3];
			Corpusbuf = (double *)malloc(nCorpusbuf*d*sizeof(double));
			muArraybuf = (double *)malloc(nodesbuf*sizeof(double));
			treeIdxbuf = (int *)malloc(3*nodes*sizeof(int));
			treeDatabuf = (int *)malloc(treeDataSizebuf*B*sizeof(int));
			/* Non-Blocking Receive */
			MPI_Irecv(Corpusbuf,nCorpusbuf*d,MPI_DOUBLE,previous,0,MPI_COMM_WORLD,&reqs[4]); 
			MPI_Irecv(muArraybuf,nodesbuf,MPI_DOUBLE,previous,0,MPI_COMM_WORLD,&reqs[5]); 
			MPI_Irecv(treeIdxbuf,nodesbuf*3,MPI_INT,previous,0,MPI_COMM_WORLD,&reqs[6]); 
			MPI_Irecv(treeDatabuf,treeDataSizebuf*B,MPI_INT,previous,0,MPI_COMM_WORLD,&reqs[7]); 
		}
		/* Init tree */
		treeReturn tree;
		tree.treeIdx=treeIdx;
		tree.muArray=muArray;
		tree.treeData=treeData;
		tree.currNode=currNode;

		int IdxStart=(Ycurrent == 0 )? 0 :displacements[Ycurrent]/k;

		/* Calculate kNNs for Y */
		for (int YIdx=0;YIdx<m;YIdx++){
			result[YIdx]=kNNSearchTree(Corpus,d,tree,B,result[YIdx],X,YIdx,IdxStart);
		}
		/* If is the last iteration skip wait */
		if (iteration < numtasks -1){
			/* Wait for the asynchrous send-receive to return */
			MPI_Waitall(8,reqs,stat);
			/* Free Y and swap Y and Z */
			free(Corpus);
			free(muArray);
			free(treeIdx);
			free(treeData);
			Corpus=Corpusbuf;
			muArray=muArraybuf;
			treeIdx=treeIdxbuf;
			treeData=treeDatabuf;
			nCorpus-nCorpusbuf;
			nodes=nodesbuf;
			treeDataSize=treeDataSizebuf;
			currNode=currNodebuf;
			Ycurrent--;
			if (Ycurrent == -1) Ycurrent = numtasks - 1 ;
		}
		
	
	}
	free(Corpusbuf);
	free(muArraybuf);
	free(treeIdxbuf);
	free(treeDatabuf);
	/* Sort the result */
	double 	*finalDist = (double *)malloc(n*k*sizeof(double));
	int 	*finalIdx = (int *)malloc(n*k*sizeof(int));
	
	double 	*partDist = (double *)malloc(m*k*sizeof(double));
	int 	*partIdx = (int *)malloc(m*k*sizeof(int));
	for (int i=0;i<m;i++){
		for(int j=0;j<k;j++){
			partDist[i*k+j]=result[i].ndist[j];
			partIdx[i*k+j]=result[i].nidx[j];
		}
		free(result[i].ndist);
		free(result[i].nidx);	
	}


	int receiveCount[numtasks];
	int countm = m*k;
	MPI_Allgather(&countm,1,MPI_INT,receiveCount,1,MPI_INT,MPI_COMM_WORLD);


	MPI_Allgatherv(partDist,m*k,MPI_DOUBLE,finalDist,receiveCount,displacements,MPI_DOUBLE,MPI_COMM_WORLD);
	MPI_Allgatherv(partIdx ,m*k,MPI_INT   ,finalIdx ,receiveCount,displacements,MPI_INT   ,MPI_COMM_WORLD);
	MPI_Finalize();
	free(partDist);
	free(partIdx);

	/* Setup result for output */
	knnresult finalres;
	finalres.k=k;
	finalres.m=n;
	finalres.nidx=finalIdx;
	finalres.ndist=finalDist;
	if(rank==0){
	for (int i = 0 ; i<m*k;i++)
		printf("dist: %lf \t index: %d  \n",finalres.ndist[i],finalres.nidx[i]);
	}
	return finalres;

}

/* kNN tree search method
 * It is assumed that the knnresult is initialized by setting
 * */
knnresult
kNNSearchTree(double *X,int d,treeReturn tree,int B,knnresult result,double *Y,int YIdx,int IdxStart){
 	/* Check if node is leaf */
	if (tree.currNode==INT_MIN){
		/* Handle empty */
		return result;
	}
	if (tree.currNode<0){
		//printf("In leaf -- currentNode %d\n",tree.currNode);
		/* For element in leaf check the distance from query */
		for (int i=0;i<B;i++){
			int elemIndex=tree.treeData[(-tree.currNode)*B+i];
			//printf("In leaf -- treeData[%d]=%d\n",(-tree.currNode)*B+i,elemIndex);
			/* if element is -1 no more data exit */
			if (elemIndex==-1) break;
			/* Calculate distance */
			double dist=calcDistance(X,Y,d,elemIndex,YIdx);
			/* if distance is smaller than farthest update the knn*/
			result = checkIfNN(dist,elemIndex,result,IdxStart);
		}
		return result;
	}
	else{	
		/* Get vantage point's Index */
		int vpIndex = tree.treeIdx[(tree.currNode-1)*3 + vpIdx];
		/* Calculate distance from query to vantage */
		double dist = calcDistance(X,Y,d,vpIndex,YIdx);
		//printf("Distance to Vantage: %f\n",dist);
		/*  Check Vantage is NN */
		result = checkIfNN(dist,vpIndex,result,IdxStart);
		/* Get mu  */
		double mu = tree.muArray[tree.currNode-1];
		/* Recursive checks */
		int currNode=tree.currNode; //Assign a currNode because it changes after returning from recursion
		//printf("CurrentNode %d vpIdx %d dist %f mu %f\n",tree.currNode,vpIndex,dist,mu);
		if(dist<mu){
			if(dist < (mu + result.ndist[result.k-1] )){
				tree.currNode = tree.treeIdx[(currNode-1)*3 + left];
				result = kNNSearchTree(X,d,tree,B,result,Y,YIdx,IdxStart);
			}
			if(dist >= (mu - result.ndist[result.k-1] )){
				tree.currNode = tree.treeIdx[(currNode-1)*3 + right];
				result = kNNSearchTree(X,d,tree,B,result,Y,YIdx,IdxStart);
			}
		}
		else{

			if(dist >= (mu - result.ndist[result.k-1] )){
				tree.currNode = tree.treeIdx[(currNode-1)*3 + right];
				result = kNNSearchTree(X,d,tree,B,result,Y,YIdx,IdxStart);
			}
			if(dist < (mu + result.ndist[result.k-1] )){
				tree.currNode = tree.treeIdx[(currNode-1)*3 + left];
				result = kNNSearchTree(X,d,tree,B,result,Y,YIdx,IdxStart);
			}
		}

		return result;
	}	
}
knnresult
checkIfNN(double dist,int elemIndex,knnresult result,int IdxStart){
	/* If distance is smaler that farthest the NN exist  */
	if (dist<=result.ndist[result.k-1] && dist>0.0000001){
		/* Iterate through current neightbours */
		for (int knn=result.k-1;knn>=0;knn--){
			/* if distance smaller that the neighbour update it  */
			if (dist <= result.ndist[knn]){
				double tmpdist = result.ndist[knn];
				int tmpidx = result.nidx[knn];
				result.ndist[knn] = dist ;
				result.nidx[knn] = elemIndex + IdxStart;
				/* If k index (knn) is 0 there is no place to put the tmp so break  */
				if (knn!=result.k-1){
					result.ndist[knn+1] = tmpdist;
					result.nidx[knn+1] = tmpidx;
				}
			}
		}
	}

	return result;

}
treeReturn
makeTree(double *X,int n,int d, int *treeIdx,double *muArray, int *treeData , int *treeDataSize, int *availableIndices, int availableIndicesSize,const int B,int *nodes){
	if (X==NULL) fprintf(stderr,"Line %d: error X is NULL",__LINE__);
	/* Declare output */
	treeReturn result;	
	/* Check if node is leaf */
	if (availableIndicesSize<=B){
		//printf("Leaf with %d data\n",availableIndicesSize);
		/* Check if data is zero the add the MIN_INT as current node */
		if(availableIndicesSize==0){
			free(availableIndices);
			result.currNode= INT_MIN;
			result.treeData=treeData;
			result.treeIdx=treeIdx;
			result.muArray=muArray;
			return result;
		}

		/* Add a new dataSize place  */
		(*treeDataSize)++;
		int currTreeDataSize = *treeDataSize;
		treeData=(int *)realloc(treeData,currTreeDataSize*B*sizeof(int));
		/* Copy Data from available Indices to treeData 
		 * if less than B exist fill blanks with -1*/
		for(int i=0;i<B;i++){
			if(i<availableIndicesSize){
				treeData[B * (currTreeDataSize-1)+i] = availableIndices[i];
			}
			else{
				treeData[B * (currTreeDataSize-1)+i] = -1;

			}
		}
		/* Free Data And setup the output */
		free(availableIndices);
		result.currNode=-(currTreeDataSize-1);
		result.treeData=treeData;
		result.treeIdx=treeIdx;
		result.muArray=muArray;
		//printf("Leaf Current Node %d\n",result.currNode);
		return result;
	}
	/* Add a new node */
	(*nodes)++;
	int currNode=*nodes;
	printf("In Node %d avilable Indices %d\n",currNode,availableIndicesSize);
	int treeEl=3*(currNode-1);
	/* Please note the (nodes *3) */
	treeIdx = (int *)realloc(treeIdx,currNode * 3*sizeof(int));
	muArray = (double *)realloc(muArray,currNode * sizeof(double));
	/* Find the Vantage point*/
	treeIdx[vpIdx + treeEl] = selectVp(X,n,d,availableIndices,availableIndicesSize);
	/* Remove the Vantage point from availableIndices */
	int passedEl=0;
	for (int i =0;i<availableIndicesSize-1;i++){
		if (availableIndices[i]==treeIdx[vpIdx + treeEl])
			passedEl=1;
		availableIndices[i]=availableIndices[i+passedEl];
	}
	/* Find distances from Vantage Point */
	double *distArray = (double *)malloc((availableIndicesSize-1)*sizeof(double));
	for (int i=0;i<availableIndicesSize-1;i++){
		distArray[i]=calcDistance(X,X,d,treeIdx[vpIdx + treeEl],availableIndices[i]);
	}
	/* Sort the distances and fid median */
	quicksort(distArray,availableIndices,0,availableIndicesSize-2);
	int median;
	if ((availableIndicesSize-1)%2==0)
		median=(availableIndicesSize-1)/2;
	else
		median=floor((availableIndicesSize-1)/2)+1;
	/* Add median to treeIdx */
	muArray[currNode-1]=distArray[median];
	/* malloc memory for left and right available indices */
	int * leftAvailableIdx = NULL;
	int * rightAvailableIdx = NULL; 
	int leftAvailableIdxSize = 0;
	int rightAvailableIdxSize = 0;
	/* Add element to left and right */
	for (int i=0;i<availableIndicesSize-1;i++){
		if (distArray[i]<muArray[currNode-1]){
			leftAvailableIdxSize++;
			leftAvailableIdx=(int *)realloc(leftAvailableIdx,sizeof(int)*leftAvailableIdxSize);
			leftAvailableIdx[leftAvailableIdxSize-1]=availableIndices[i];
		}else{
			rightAvailableIdxSize++;
			rightAvailableIdx=(int *)realloc(rightAvailableIdx,sizeof(int)*rightAvailableIdxSize);
			rightAvailableIdx[rightAvailableIdxSize-1]=availableIndices[i];
		}
	}
	/* Free distArray and availableIndices */
	free(distArray);
	free(availableIndices);
	/* Recursively call the function again */
	/*  For the left branch */
	treeReturn leftreturn =	makeTree(X,n,d,treeIdx,muArray,treeData,treeDataSize,leftAvailableIdx,leftAvailableIdxSize,B,nodes);
	/* Update the treeIdx and the treeData pointers chenged from realloc */
	treeIdx=leftreturn.treeIdx;
	treeData=leftreturn.treeData;
	muArray=leftreturn.muArray;
	/*  For the right branch*/
	treeReturn rightreturn = makeTree(X,n,d,treeIdx,muArray,treeData,treeDataSize,rightAvailableIdx,rightAvailableIdxSize,B,nodes);
	treeIdx=rightreturn.treeIdx;
	treeData=rightreturn.treeData;
	muArray=rightreturn.muArray;
	/* Fill the treeIdx left and right*/
	treeIdx[left+treeEl]=leftreturn.currNode;
	treeIdx[right+treeEl]=rightreturn.currNode;
	/* Setup output */
	result.currNode=currNode;
	result.treeData=treeData;
	result.treeIdx=treeIdx;
	result.muArray=muArray;
	
	return result;


}
int
selectVp(double *X,int n,int d,int *availableIndices,int availableIndicesSize){
	# define sampleCoeff 0.05
	/*  Get seed for random*/
	time_t t;
	srand((unsigned) time(&t));
	/* Calc the 2 samples sizes */
	int AsampleSize = (int) ((double)availableIndicesSize*sampleCoeff);
	int BsampleSize= (int) ((double)availableIndicesSize*sampleCoeff*0.25);
	if (AsampleSize==0) AsampleSize=1;
	if (BsampleSize==0) BsampleSize=1;
	/* Allocate memories for sample arrays */
	int *isASampleDrawed = (int *)calloc(availableIndicesSize,sizeof(int));
	int *AsampleIdx = (int *)malloc(AsampleSize*sizeof(int));
	int *BsampleIdx = (int *)malloc(BsampleSize*sizeof(int));
	double bestSpread=0,bestp;
	/* Select a random sample of unique indices from available indices */
	/* Iterate for every sample from setA */
	for (int i=0 ; i<AsampleSize;i++){
		/* Unconditional loop */
		while(1){
			/* Get a random number in the range [0,availableIndicesSize] */
			int draw = rand() % availableIndicesSize;
			/* If sample was not drawn before */
			if (isASampleDrawed[draw]==0){
				/* Add Index to Asample set */
				AsampleIdx[i]=availableIndices[draw];
				/* And set the Index as drawn */
				isASampleDrawed[draw]=1;
				break;
			}
		}
	}
	/* Iterate through the points found in A set  */
	for (int p=0; p<AsampleSize;p++){
		/* Variables */
		int *isBSampleDrawed = (int *)calloc(availableIndicesSize,sizeof(int));
		double distArray[BsampleSize];
		long double spread=0;
		int median=0;
		/* Select a random sample of unique indices for set B from available indices */
		for (int i=0 ; i<BsampleSize;i++){
			while(1){
				int draw = rand() % availableIndicesSize;
				if (isASampleDrawed[draw]==0 && isBSampleDrawed[draw]==0){
					BsampleIdx[i]=availableIndices[draw];
					isBSampleDrawed[draw]=1;
					break;
				}
			}
		}
		/* Fill distance array */
		for (int bsample=0;bsample<BsampleSize;bsample++){
			distArray[bsample]=calcDistance(X,X,d,AsampleIdx[p],BsampleIdx[bsample]);
		}
		/* Sort the results  */
		quicksort(distArray,BsampleIdx,0,BsampleSize-1);
		/* Get the median value index */
		if (BsampleSize%2==0)
			median=BsampleSize/2;
		else
			median=floor(BsampleSize/2)+1;
		if (BsampleSize<3)
			median=1;
		/* Calculate spread */
		for (int i=0;i<BsampleSize;i++)
			spread+=(distArray[i]-distArray[median])*(distArray[i]-distArray[median]);
		spread=sqrt(spread/BsampleSize);
		/* If the spread is best keep the point as a vantage */
		if (spread>bestSpread){
			bestSpread=spread;
			bestp=AsampleIdx[p];
		}
		free(isBSampleDrawed);		
	}
	free(isASampleDrawed);
        free(AsampleIdx);
        free(BsampleIdx);

	return bestp;	
}
/* Calculate Distance between point A and B given a database Array X ,dimensions d and two Indices of X AIdx and BIdx*/
double
calcDistance(double *X,double *Y,int d,int AIdx,int BIdx){
	double temp=0;
	for (int i=0;i<d;i++){
		temp+=(X[(d*AIdx)+i] -  Y[(d*BIdx)+i])*(X[(d*AIdx)+i] -  Y[(d*BIdx)+i]);
	}
	return sqrt(temp);
}

void 
quicksort(double *distance,int * idxArray,int first,int last){
   int i, j, pivot,idxtemp; 
   double temp;

   if(first<last){
      pivot=first;
      i=first;
      j=last;

      while(i<j){
         while(distance[i]<=distance[pivot]&&i<last)
            i++;
         while(distance[j]>distance[pivot])
            j--;
         if(i<j){
            temp=distance[i];
            idxtemp=idxArray[i];
            distance[i]=distance[j];
            idxArray[i]=idxArray[j];
            distance[j]=temp;
            idxArray[j]=idxtemp;
         }
      }

      temp=distance[pivot];
      idxtemp=idxArray[pivot];
      distance[pivot]=distance[j];
      idxArray[pivot]=idxArray[j];
      distance[j]=temp;
      idxArray[j]=idxtemp;
      quicksort(distance,idxArray,first,j-1);
      quicksort(distance,idxArray,j+1,last);

   }
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
