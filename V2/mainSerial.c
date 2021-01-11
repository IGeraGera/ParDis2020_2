#include<stdio.h>
#include<stdlib.h>
#include<errno.h>
#include<string.h>
#include<math.h>
#include<float.h>
#include<time.h>
#include<limits.h>
#include<cblas.h>

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

/*!

  \param  X      Corpus data points              [n-by-d]
  \param  n      Number of corpus points         [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/

/* Functions Declaration */
knnresult kNNSearchTree(double *X,int d,treeReturn tree,int B,knnresult result,double *Y,int YIdx);
knnresult checkIfNN(double dist,int elemIndex,knnresult result);
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
	/* kNN variables*/
	int n = dataRows;
	int d = attributes;
	int k = 3;
	printf("Dimensions %d Rows %d \n",d,n);
	double *X = (double *)malloc(n*d*sizeof(double));
	/* Init X */
	for (int i= 0 ; i<n; i++){
		for (int j=0; j<d ; j++){
			X[d*i+j]=data[i][j];
		}
	}
	/* Free  data memory*/
	while(dataRows) free(data[--dataRows]);
	free(data);	
	/* Tree Indices  */
	/* [ vpIdx left right | vpIdx left right .... | vpIdx left right ]*/
	#define vpIdx 0
	#define left 1
	#define right 2
	int *treeIdx = (int *)malloc(0* sizeof(int));
	/* Num of elements in leaves B */
	const int B = 2;
	/* Data array */
	int treeDataSize =1;
	int *treeData = (int *)malloc(B*sizeof(int));
	for(int i=0;i<B;i++)
		treeData[i]=-1;
	/* Num of Nodes*/
	int nodes=0;
	/* Avilable Indices matrix and Init it*/
	int *availableIndices = (int *)malloc(n*sizeof(int));
	for (int i=0;i<n;i++)
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
	puts("Making tree");
	treeReturn tree = makeTree(X,n,d,treeIdx,muArray,treeData,&treeDataSize,availableIndices,n,B,&nodes);
	treeIdx = tree.treeIdx;
	treeData = tree.treeData;
	muArray = tree.muArray;
	/* Set up the knnresult struct */
	knnresult result;
	result.k=k;
	result.nidx = (int *)malloc(k*sizeof(int));
	result.ndist = (double *)malloc(k*sizeof(double));
	for(int i=0;i<k;i++){
		result.ndist[i]=DBL_MAX;
		result.nidx[i]=-1;
	}

	int YIdx =0;
	kNNSearchTree(X,d,tree,B,result,X,YIdx);
	for(int i=0;i<k;i++){
		printf("k: %d dist: %f index: %d\n",k,result.ndist[i],result.nidx[i]);
	}
	
	/* Free Data Memory */
	free(treeIdx);
	free(treeData);
	free(muArray);
	free(result.nidx);
	free(result.ndist);

	/*
	struct timeval timeStart,timeEnd;
	double totaltime;
	gettimeofday(&timeStart,NULL);
	//knnresult res=kNN(X,Y,n,m,d,k);
	gettimeofday(&timeEnd,NULL);
	totaltime = (timeEnd.tv_sec*1000 + timeEnd.tv_usec/1000) - (timeStart.tv_sec*1000 + timeStart.tv_usec/1000) ;
	*/
	/* Print output */
	//for (int i = 0 ; i<m*k;i++)
	//	printf("dist: %lf \t index: %d  \n",res.ndist[i],res.nidx[i]);

	//printf("Total time : %.4f ms\n",totaltime);
	/*Ending Proccess deallocate memory  */
	free(X);
	fclose(fp);
	exit(EXIT_SUCCESS);

}
/* kNN tree search method
 * It is assumed that the knnresult is initialized by setting
 * */
knnresult
kNNSearchTree(double *X,int d,treeReturn tree,int B,knnresult result,double *Y,int YIdx){
 	/* Check if node is leaf */
	if (tree.currNode==INT_MIN){
		/* Handle empty */
		return result;
	}
	if (tree.currNode<0){
		printf("In leaf -- currentNode %d\n",tree.currNode);
		/* For element in leaf check the distance from query */
		for (int i=0;i<B;i++){
			int elemIndex=tree.treeData[(-tree.currNode)*B+i];
			printf("In leaf -- treeData[%d]=%d\n",(-tree.currNode)*B+i,elemIndex);
			/* if element is -1 no more data exit */
			if (elemIndex==-1) break;
			/* Calculate distance */
			double dist=calcDistance(X,Y,d,elemIndex,YIdx);
			/* if distance is smaller than farthest update the knn*/
			result = checkIfNN(dist,elemIndex,result);
		}
		return result;
	}
	else{	
		/* Get vantage point's Index */
		int vpIndex = tree.treeIdx[(tree.currNode-1)*3 + vpIdx];
		/* Calculate distance from query to vantage */
		double dist = calcDistance(X,Y,d,vpIndex,YIdx);
		printf("Distance to Vantage: %f\n",dist);
		/*  Check Vantage is NN */
		result = checkIfNN(dist,vpIndex,result);
		/* Get mu  */
		double mu = tree.muArray[tree.currNode-1];
		/* Recursive checks */
		int currNode=tree.currNode; //Assign a currNode because it changes after returning from recursion
		printf("CurrentNode %d vpIdx %d dist %f mu %f\n",tree.currNode,vpIndex,dist,mu);
		if(dist<mu){
			printf("1dist<mu %f %f\n",dist,(mu + result.ndist[result.k-1] ));
			if(dist < (mu + result.ndist[result.k-1] )){
				tree.currNode = tree.treeIdx[(currNode-1)*3 + left];
		printf("NextNode1 %d\n",tree.currNode);
				result = kNNSearchTree(X,d,tree,B,result,Y,YIdx);
			}
			printf("2dist>mu %f %f\n",dist,(mu + result.ndist[result.k-1] ));
			if(dist >= (mu - result.ndist[result.k-1] )){
				tree.currNode = tree.treeIdx[(currNode-1)*3 + right];
		printf("NextNode2 %d\n",tree.currNode);
				result = kNNSearchTree(X,d,tree,B,result,Y,YIdx);
			}
		}
		else{
			printf("3dist>mu %f %f\n",dist,(mu + result.ndist[result.k-1] ));

			if(dist >= (mu - result.ndist[result.k-1] )){
				tree.currNode = tree.treeIdx[(currNode-1)*3 + right];
		printf("NextNode3 %d\n",tree.currNode);
				result = kNNSearchTree(X,d,tree,B,result,Y,YIdx);
			}
			printf("4dist>mu %f %f\n",dist,(mu + result.ndist[result.k-1] ));
			if(dist < (mu + result.ndist[result.k-1] )){
				tree.currNode = tree.treeIdx[(currNode-1)*3 + left];
		printf("NextNode4 %d\n",tree.currNode);
				result = kNNSearchTree(X,d,tree,B,result,Y,YIdx);
			}
		}

		return result;
	}	
}
knnresult
checkIfNN(double dist,int elemIndex,knnresult result){
	/* If distance is smaler that farthest the NN exist  */
	if (dist<=result.ndist[result.k-1] && dist>0.0000001){
		/* Iterate through current neightbours */
		for (int knn=result.k-1;knn>=0;knn--){
			/* if distance smaller that the neighbour update it  */
			if (dist <= result.ndist[knn]){
				double tmpdist = result.ndist[knn];
				int tmpidx = result.nidx[knn];
				result.ndist[knn] = dist ;
				result.nidx[knn] = elemIndex ;
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
		printf("Leaf with %d data\n",availableIndicesSize);
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
		printf("Leaf Current Node %d\n",result.currNode);
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
