load Data_PTC_vs_FTC.mat
X = Data.X;
D = Data.D;
rowNames = Data.gene_names;
x_D = [D;X];



TP = 65;
TN = 18;
FN = 2;
FP = 1;

Acc = (TP+TN)/(TP+TN+FP+FN)
Err = 1 - Acc
Spec = TN/(TN+FP)
Sens = TP/(TP+FN)