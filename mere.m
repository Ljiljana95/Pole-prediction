
function [Tacnost,Preciznost,Odziv,F,FPR] = mere(t)

    TP= t(1,1);
    FN= t(1,2);
    FP= t(2,1);
    TN= t(2,2);
%racunanje mera
    Tacnost = (TP+TN)/(TP+TN+FP+FN);
    Preciznost = TP/(TP+FP);
    Odziv = TP/(TP+FN);
    F= 2/(1/Odziv + 1/Preciznost);
    FPR= FP/(FP+TN);

end