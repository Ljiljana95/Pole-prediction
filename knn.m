%PROJEKAT IZ PREPOZNAVANJA OBLIKA
    % Predikcija pola
% Stojanovic Ivana EE 59/2014 
% Ljiljana Popovic EE 72/2014
 
function knn(test,trening,test_l,tren_l,k) 

 format short
 disp('*** KNN ***')
 
 cv= cvpartition(tren_l,'KFold',5); % krosvalidacija KFold 5
 for i = 1 : k
    t=[0 0; 0 0];
        for j = 1: 5
            % uzivanje 4 fold-a za trening i 1 za test dobijeni
            % krosvalidacijom
            TTe= cv.test(j);
            TTr= cv.training(j);
            TTE = trening(TTe==1,:);
            TTE_L =tren_l(TTe==1,:);
            TTR = trening(TTr==1,:);
            TTR_L =tren_l(TTr==1,:);
            % pravljenje modela
            model = fitcknn(TTR,TTR_L,'NumNeighbors',i,'NSMethod','exhaustive','Distance','euclidean','BreakTies','nearest','Standardize',true);
         
            %predikcija
            TTE_pred_L= predict(model,TTE);
            % matrica konfuzije
            a = confusionmat(TTE_L,TTE_pred_L');
            t=[a(1,1)+t(1,1), a(1,2)+t(1,2); a(2,1)+t(2,1), a(2,2)+t(2,2)];
            
        end
        
   Tacnost(i)= mere(t);
 end

[ma,ind]= max(Tacnost); 

%obuka krajnjeg modela
model_kraj = fitcknn(trening,tren_l,'NumNeighbors',ind,'NSMethod','exhaustive','Distance','euclidean','BreakTies','nearest','Standardize',true);
cmodel_kraj= crossval(model_kraj,'KFold',5);
[labelsOut,scoreOut] = kfoldPredict(cmodel_kraj);
[Xx,Yy, Th, AUC] = perfcurve(tren_l,scoreOut(:,cmodel_kraj.ClassNames),true);
figure, plot(Xx,Yy,'--')
title('ROC kriva; KNN')
xlabel('False positive rate')
ylabel('True positive rate')
% predicija nad test podacima
pred_labele = predict(model_kraj, test);
% matrica kofuzije
[C,order] = confusionmat(test_l,pred_labele')

[T,P,O,F,FPR]=mere(C);
 
Q= table(T,P,O,F,'RowNames',{['k=' num2str(ind)]},'VariableNames',{'Tacnost', 'Preciznost', 'Odziv','F_mera'})
end