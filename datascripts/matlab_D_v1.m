

# Get data
data = importdata("../Datasets/Container_Crane_Controller_Data_Set.csv");
#data = importdata("../Datasets/BreastCancerCoimbra.csv");
n =size(data)(1);
d =size(data)(2);


Y=data(4,:);
X=data;

D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).')

sort(D)(2:6)