

# Get data
#data = importdata("../Datasets/Container_Crane_Controller_Data_Set.csv");
data = importdata("../Datasets/BreastCancerCoimbra.csv");
dataRows =size(data)(1);
d =size(data)(2);
n=floor(dataRows*0.75);
m=dataRows-n;

X=data(1:n,:);
Y=data(n+1:end,:);


D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).')
