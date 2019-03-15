%PROJEKAT IZ PREPOZNAVANJA OBLIKA
    % Predikcija pola
% Stojanovic Ivana EE 59/2014 
% Ljiljana Popovic EE 72/2014
 
function random(test,trening,test_l,tren_l,k) 

 format short
 disp('*** Random Forest ***')
 
            b=TreeBagger(k,trening,tren_l,'oobpred','on');
            figure,plot(oobError(b))
            xlabel('broj stabala')
            ylabel('out-of-bag greska')
            Err=oobError(b);
            [A,scoreOut]=oobPredict(b);
            TTE_pred_L= predict(b,test);
            TTE_pred_L= double(cell2mat(TTE_pred_L))-48;
            
            [Xx,Yy, Th, AUC] = perfcurve(tren_l,scoreOut(:,logical([0;1])),true);
            figure, plot(Xx,Yy,'--')
            title('ROC kriva; Random forest')
            xlabel('False positive rate')
            ylabel('True positive rate')
            
            C = confusionmat(test_l,logical(TTE_pred_L)')
            [T,P,O,F,FPR]=mere(C);
 
           Q= table(T,P,O,F,'RowNames',{'Random forest'},'VariableNames',{'Tacnost', 'Preciznost', 'Odziv','F_mera'})
end